import unittest
from pathlib import Path

from soma_voice import speech_helper_path, split_speakable


class VoiceTests(unittest.TestCase):
    def test_streaming_speech_waits_for_sentence_boundaries(self):
        phrases, tail = split_speakable(
            "hello, james. how are you? i am still")
        self.assertEqual(phrases, ["hello, james.", "how are you?"])
        self.assertEqual(tail, "i am still")

    def test_newline_is_a_speech_boundary(self):
        phrases, tail = split_speakable("first line\nsecond")
        self.assertEqual(phrases, ["first line"])
        self.assertEqual(tail, "second")

    def test_development_helper_is_local_executable(self):
        helper = speech_helper_path()
        self.assertIsNotNone(helper)
        self.assertEqual(Path(helper).name, "soma-speech")


if __name__ == "__main__":
    unittest.main()
