'''Synthetic datasets for language modeling.'''
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

TensorDataLoader = DataLoader[tuple[torch.Tensor, ...]]

class Vocab:
    """Custom vocab."""
    def __init__(self, vocab_size: int, special_vocabs: dict[str, str]):
        # Special tokens hold copy_prefix and pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs
        vocab_size = vocab_size - len(special_vocabs)

        print(f"Vocab size excluding special vocab: {vocab_size}")
        print(f"Special vocabs size: {len(special_vocabs)}")
        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.vocab.append('-100')
        self.v2id = {v:i for i,v in enumerate(self.vocab)}
        self.v2id['-100'] = -100
        self.vocab_size = len(vocab)

    def get_next_vocab(self, token: str) -> str:
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def copy_prefix(self) -> str:
        return self.special_vocabs["copy_prefix"]

    @property
    def special_tokens(self) -> set[str]:
        return set(self.special_vocabs.values())

    def get_id(self, token: str) -> int:
        return self.v2id[token]

    def get_vocab(self, id: int) -> str:
        return self.vocab[id]

    def __len__(self) -> int:
        return len(self.vocab)


class Tokenizer:
    """Custom Tokenizer for our own vocab."""
    def __init__(self, vocab: Vocab, max_length: int = -1, 
                 len_label_tokens: int = 1, len_copy_tokens: int = 1):
        self.vocab = vocab
        self.max_length = max_length
        self.len_label_tokens = len_label_tokens
        self.len_copy_tokens = len_copy_tokens

    def tokenize(self, text: str, return_tensor: bool = False, 
                 mask_input: bool = False) -> dict[str, torch.LongTensor | list[int]]:
        input_ids = [self.vocab.get_id(t) for t in text.split()]
        if self.vocab.get_id(self.vocab.copy_prefix) not in input_ids:
            raise ValueError("Input text must contain copy_prefix token.")
        copy_prefix_pos = input_ids.index(self.vocab.get_id(self.vocab.copy_prefix))
        labels = input_ids
        if mask_input:
            labels = [-100] * (copy_prefix_pos+1) + labels[copy_prefix_pos+1:]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.vocab.get_vocab(id) for id in ids])


def base_ar(vocab: Vocab, input_seq_len: int, rng: random.Random) -> str:
    """Generate sequence where the input has a sequence of key value pairs
    and the copy prefix at the end, and then a key value pair is inserted
    after the copy prefix."""
    non_special_vocab_size = len(vocab.non_special_vocab) 
    keys = vocab.non_special_vocab[:non_special_vocab_size // 2]
    values = vocab.non_special_vocab[non_special_vocab_size // 2:]
    kv_map = {k: rng.choice(values) for k in keys}

    key_present: set[str] = set()
    vocab_seq: list[str] = []
    for _ in range(input_seq_len // 2):
        k = rng.choice(keys)
        vocab_seq += [k, kv_map[k]]
        key_present.add(k)

    # add key value pair that has already been seen before
    k = rng.choice(list(key_present))
    vocab_seq += [vocab.copy_prefix, k, kv_map[k]]
    return " ".join(vocab_seq)


def generate_ar_dataset(
        num_examples: int, 
        num_test_examples: int,
        input_seq_len: int, 
        tokenizer: Tokenizer, 
        vocab: Vocab, 
        rng: random.Random,
        ignore_train: bool = False,
    ) -> dict[str, TensorDataset]:
    train_tensor: torch.Tensor | None = None
    test_tensor: torch.Tensor | None = None
    all_examples: list[torch.LongTensor] = []
    num_extra_seq_len = 2
    if train_tensor is None or test_tensor is None: 
        for example_count, name in [(num_examples, "training"), (num_test_examples, "test")]:
            examples: list[list[int]] = []
            for _ in tqdm(range(example_count), f"Generating {name} dataset", unit=" examples"):
                vocab_seq = base_ar(vocab, input_seq_len, rng)
                example = tokenizer.tokenize(vocab_seq, return_tensor=False)['input_ids']
                examples.append(example)
            rng.shuffle(examples)
            all_examples.append(torch.LongTensor(examples))
        train_tensor = torch.stack([
            torch.stack([
                example[:-1], 
                example[1:]
            ]) for example in all_examples[0]])
        test_tensor = torch.stack([
            torch.stack([
                example[:-1],
                example[1:]
            ]) for example in all_examples[1]])
        if ignore_train:
            train_tensor[:, 1, :-1 * (num_extra_seq_len - 1)] = -100
        test_tensor[:, 1, :-1 * (num_extra_seq_len - 1)] = -100
    dataset = {
        'train': TensorDataset(train_tensor[:, 0, :], train_tensor[:, 1, :]),
        'test': TensorDataset(test_tensor[:, 0, :], test_tensor[:, 1, :])
    }
    return dataset


class ICLDataModule:
    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        vocab_size: int,
        input_seq_len: int,
        seed: int = 0,
        batch_size: int = 32,
        ignore_train: bool = False,
        **dataset_kwargs,
    ):
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.input_seq_len = input_seq_len
        self.vocab_size = vocab_size
        self.ignore_train=ignore_train
        self.seed = seed
        self.batch_size = batch_size

        # Special Tokens
        special_vocabs = {
            "copy_prefix": "=>",
        }
        self.special_vocabs = special_vocabs   
        self.vocab = Vocab(
            vocab_size, 
            special_vocabs=special_vocabs, 
        )
        self.tokenizer = Tokenizer(
            self.vocab, 
            max_length=self.input_seq_len,
        )

    def setup(self):
        self.rng = random.Random(self.seed)
        random.seed(self.seed)
        dataset = generate_ar_dataset(
            num_examples=self.num_examples, 
            num_test_examples=self.num_test_examples,
            input_seq_len=self.input_seq_len,
            tokenizer=self.tokenizer, 
            vocab=self.vocab,
            rng=self.rng,
            ignore_train=self.ignore_train,
        )
        self.dataset = dataset        

    def train_dataloader(self, *args, **kwargs) -> TensorDataLoader:
        return self._data_loader(self.dataset['train'], shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> TensorDataLoader:
        return self._data_loader(self.dataset['test'], shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> TensorDataLoader:
        return self._data_loader(self.dataset['test'], shuffle=False)

    def _data_loader(self, dataset: TensorDataset, shuffle: bool = False) -> TensorDataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=shuffle,
            persistent_workers=True
        )
    
