import neuspell
from nltk.tokenize import WordPunctTokenizer


class SpellChecker:
    """
    A class that wraps around neuspell (https://github.com/neuspell) and provides it with a fine-tuned bert spellchecker
    """

    def __init__(self, checker=neuspell.BertChecker(), model_path: str = "fine_tuned_model", delete_limit: int = 0):
        """
        init:
        Parameters
        ----------
        checker: type of the fine tuned checker we will be feeding the wrapper, default = BertChecker
        model_path: path to the model files
        delete_limit: this limit defines the amount of tokens we let the spellchecker delete, default = 0
        """
        self.checker = checker
        self.model = self.checker.from_pretrained(ckpt_path=model_path)
        self.del_lim = delete_limit
        self.tokenizer = WordPunctTokenizer()
        try:
            checker.correct("This is a test string")
        except AssertionError:
            print("The checker did not load correctly. Please make sure that all necessary files are in the "
                  "/fine_tuned_model/ directory")

    def check(self, input_str: str):
        """
        Method that takes in a string and runs it through the spellchecker. If the spellchecker deletes more tokens than
        allowed by self.del_lim, this returns the initial input.

        Parameters
        ----------
        input_str: a string of text

        Returns a string
        -------

        """
        input_len = len(self.tokenizer.tokenize(input_str))
        output = self.checker.correct(input_str)
        output_len = len(self.tokenizer.tokenize(output))
        if input_len - output_len > self.del_lim:
            return input_str
        else:
            return output

