import os
import torch
import json
import boto3
import tempfile
from torch.utils.data import Dataset, DataLoader
from botocore.exceptions import ClientError
from torch.utils.data._utils.collate import default_collate
import bisect
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

B2_ACCESS_KEY_ID = os.getenv("B2_ACCESS_KEY_ID")
B2_SECRET_ACCESS_KEY = os.getenv("B2_SECRET_ACCESS_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")
B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL")

#output_prefix = "top300uspods_tokenized"  # S3 prefix/folder for transcriptions

def empty_filter_collate_fn(batch):
    # Filter out any items where batch elements are None
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return None
    else:
        return default_collate(batch)

def get_words_between(start_time, end_time, data):
    """
    Returns all words in the transcript between start_time and end_time.

    :param start_time: The starting timestamp (inclusive).
    :param end_time: The ending timestamp (inclusive).
    :param data: The transcript data in JSON format.
    :return: A list of words between the specified timestamps.
    """
    words = []
    for segment in data.get('segments', []):
        for word_info in segment.get('words', []):
            word_start = word_info.get('start')
            word_end = word_info.get('end')
            word = word_info.get('word')

            # Skip words without timing information
            if word_start is None or word_end is None:
                continue

            # Check if the word overlaps with the time interval
            if word_end >= start_time and word_start <= end_time:
                words.append(word)
    return words

def get_words_before(timestamp, x, data):
    """
    Given a timestamp, returns the previous x words before that timestamp.

    :param timestamp: The timestamp to look up.
    :param x: The number of words to retrieve before the timestamp.
    :param data: The transcript data in JSON format.
    :return: A list of words before the specified timestamp.
    """
    all_words = []
    word_times = []

    # Flatten all words with their start times into a list
    for segment in data.get('segments', []):
        for word_info in segment.get('words', []):
            word_start = word_info.get('start')
            word = word_info.get('word')

            # Skip words without timing information
            if word_start is None:
                continue

            all_words.append((word_start, word))
            word_times.append(word_start)

    # Ensure the words are sorted by their start times
    sorted_words = sorted(zip(word_times, all_words), key=lambda w: w[0])
    word_times = [w[0] for w in sorted_words]
    all_words = [w[1][1] for w in sorted_words]

    # Find the insertion point for the timestamp
    idx = bisect.bisect_left(word_times, timestamp)

    # Get the previous x words
    start_idx = max(0, idx - x)
    words_before = all_words[start_idx:idx]
    return words_before

class B2AudioDataset(Dataset):
    def __init__(self, tokenizer, block_size, context_split_index=1024, samples_per_audio_token=192, sample_rate=44_000):
        self.block_size = block_size
        self.transcripts = self.list_in_b2('top300uspods_transcriptions/', '.json')
        self.tokenized_audio = self.list_in_b2('top300uspods_tokenized/', '.npy')
        assert len(self.transcripts)==len(self.tokenized_audio)
        self.total_files = len(self.tokenized_audio)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=B2_ACCESS_KEY_ID,
            aws_secret_access_key=B2_SECRET_ACCESS_KEY,
        )
        self.context_split_index = context_split_index # index of the first audio token
        self.samples_per_audio_token = samples_per_audio_token
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    '''
    def list_audio_files_in_b2(self):
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix='top300uspods/')
        audio_files = []

        for page in tqdm(page_iterator, desc="Listing audio files"):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith('.mp3'):
                        transcription_key = f"{output_prefix}/{key}.json"
                        if transcription_key not in self.transcription_keys:
                            audio_files.append(key)

        # Distribute files among ranks
        audio_files.sort()
        audio_files = [f for i, f in enumerate(audio_files) if i % self.world_size == self.rank]
        return audio_files
        '''

    def list_in_b2(self, prefix, suffix): #'.mp3' 'top300uspods/'
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix=prefix)

        def process_page(page):
            files = []
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith(suffix):
                        files.append(f"{prefix}/{key}.json")
            return files

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_page, page_iterator), desc="Listing audio files"))

        # Flatten list of lists
        files = [key for sublist in results for key in sublist]

        files.sort()
        return files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        tokens_s3_key = self.audio_files[idx]
        transcript_s3_key = self.audio_files[idx]
        # Download the audio file to a temporary local file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
            temp_tokens_path = tmp_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
            temp_transcript_path = tmp_file.name

        try:
            self.s3_client.download_file(Bucket=B2_BUCKET_NAME, Key=tokens_s3_key, Filename=temp_tokens_path)
            self.s3_client.download_file(Bucket=B2_BUCKET_NAME, Key=transcript_s3_key, Filename=temp_transcript_path)

            audio_tokens = np.load(temp_tokens_path)
            start_index = np.random.randint(0, len(audio_tokens) - (self.block_size-self.context_split_index+2))
            selected_audio_tokens = audio_tokens[start_index : start_index + self.block_size-self.context_split_index]
            selected_audio_tokens = torch.tensor(selected_audio_tokens)

            with open(temp_transcript_path, 'r') as file:
                transcript = json.load(file)
            words_before_audio_start = get_words_before((start_inx*self.samples_per_audio_token)/self.sample_rate, 100_000, transcript) # high value for x to get all words
            words_before_audio_start = " ".join(words_before_audio_start)
            word_tokens = self.tokenizer.encode(words_before_audio_start, bos=False, eos=False)

            #torch.cat(torch.tensor(word_tokens[len(word_tokens)-self.:]), selected_audio_tokens)
            if len(word_tokens) > self.context_split_index:
                word_tokens = word_tokens[self.context_split_index:]
            else:
                padding_length = self.context_split_index - len(word_tokens)
                word_tokens += [self.tokenizer.pad_token_id] * padding_length
            word_tokens = torch.tensor(word_tokens, dtype=torch.long)
            tokens = torch.cat([word_tokens, selected_audio_tokens], dim=0)

            x = tokens[:-1] # inputs
            y = tokens[1:] # targets
            return x, y
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return None, None
        finally:
            if temp_tokens_path and os.path.exists(temp_tokens_path):
                os.remove(temp_tokens_path)
            if temp_transcript_path and os.path.exists(temp_transcript_path):
                os.remove(temp_transcript_path)
