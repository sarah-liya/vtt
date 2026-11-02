"""

voAIce Speech Buddy

Local microphone -> Wav2Vec2 phoneme recognition -> CMUdict ARPAbet reference -> phoneme comparison.



CSV format:

    level,target_word

    1,rabbit

    1,apple

    1,banana



Important:

\- Audio is processed locally in memory. This script does not save or upload audio.

\- The model is a general phoneme-recognition model, not specifically trained for

  children aged 4-6. Scores are experimental practice scores, not clinical results.

\- TTS is intentionally excluded; connect your teammate's TTS separately.

"""



import csv

import random

import sys

import time

from pathlib import Path



import numpy as np

import sounddevice as sd

import torch
import cmudict

from transformers import (

    Wav2Vec2FeatureExtractor,

    Wav2Vec2ForCTC,

    Wav2Vec2PhonemeCTCTokenizer,

)





# ------------------------- SETTINGS -------------------------



BASE_DIR = Path(__file__).resolve().parent

WORDS_FILE = BASE_DIR / "mock_words.csv"



MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"



SAMPLE_RATE = 16000

CHANNELS = 1

BLOCK_SIZE = 1024



CALIBRATION_SECONDS = 0.4

START_TIMEOUT_SECONDS = 6.0

SILENCE_SECONDS = 2.0

MAX_RECORD_SECONDS = 8.0

MIN_AUDIO_SECONDS = 0.30



# Microphone input is int16. Adjust MIN_SPEECH_RMS if your mic is very quiet.

MIN_SPEECH_RMS = 100

NOISE_MULTIPLIER = 2.5

PRE_ROLL_SECONDS = 0.20

MAX_EXTRA_PHONES = 6





# ------------------------- WORD LIST -------------------------



def load_words():

    if not WORDS_FILE.exists():

        raise FileNotFoundError(

            f"Cannot find {WORDS_FILE.name}. Put it in the same folder as this script."

        )



    with WORDS_FILE.open("r", newline="", encoding="utf-8-sig") as f:

        rows = list(csv.DictReader(f))



    if not rows:

        raise ValueError("mock_words.csv is empty.")



    required = {"level", "target_word"}

    if not required.issubset(rows[0].keys()):

        raise ValueError("CSV must contain these columns: level,target_word")



    words = []

    for row in rows:

        word = (row.get("target_word") or "").strip().lower()

        if not word:

            continue

        try:

            level = int(row["level"])

        except (TypeError, ValueError):

            raise ValueError(f"Invalid level for word: {word}")

        words.append({"level": level, "target_word": word})



    if not words:

        raise ValueError("No valid target words were found in the CSV.")



    return words





# ------------------------- MODEL -------------------------



def load_model():

    print(f"\nLoading model: {MODEL_ID}")

    print("The first run downloads approximately 1.2 GB. Please wait.")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # This is the model's actual phoneme tokenizer, not the generic text tokenizer.

    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(

        MODEL_ID,

        do_phonemize=False,

    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)



    model.to(device)

    model.eval()



    if tokenizer.pad_token_id is None:

        raise RuntimeError("Model tokenizer has no CTC blank/pad token.")



    print(f"Model loaded on: {device}")

    return model, tokenizer, feature_extractor, device





# ------------------------- EXPECTED PRONUNCIATION: CMUDICT -------------------------

# CMUdict stores pronunciations in ARPAbet. This table maps each ARPAbet
# phone to the corresponding IPA-like symbol used as input to the Wav2Vec2
# phoneme tokenizer. The symbols are checked against the model vocabulary.
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def arpabet_to_model_symbols(arpabet_phones):
    """Convert CMUdict ARPAbet phones to an IPA-like sequence; remove stress digits."""
    ipa_phones = []
    for phone in arpabet_phones:
        base_phone = phone.rstrip("012")
        ipa_phone = ARPABET_TO_IPA.get(base_phone)
        if ipa_phone is None:
            raise ValueError(f"Unsupported CMUdict phone: {phone}")
        ipa_phones.append(ipa_phone)
    return ipa_phones


def make_expected_pronunciations(word, tokenizer):
    """
    Return all dictionary pronunciations for a word.
    Each result contains its ARPAbet form, IPA-like form and model token IDs.
    """
    entries = cmudict.dict().get(word.lower())
    if not entries:
        raise LookupError(
            f"'{word}' was not found in CMU Pronouncing Dictionary. "
            "Please choose another practice word."
        )

    candidates = []
    seen = set()
    for arpabet in entries:
        ipa_phones = arpabet_to_model_symbols(arpabet)
        ipa_text = " ".join(ipa_phones)
        if ipa_text in seen:
            continue
        seen.add(ipa_text)

        encoded = tokenizer(
            ipa_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids = list(encoded["input_ids"])
        if not ids:
            continue

        unk_id = tokenizer.unk_token_id
        if unk_id is not None and unk_id in ids:
            # The model tokenizer does not know one or more mapped symbols.
            continue

        model_phones = tokenizer.convert_ids_to_tokens(ids)
        candidates.append({
            "arpabet": " ".join(arpabet),
            "ipa": ipa_text,
            "phones": model_phones,
        })

    if not candidates:
        raise ValueError(
            f"CMUdict found '{word}', but its ARPAbet-to-IPA conversion "
            "contains symbols unsupported by the Wav2Vec2 tokenizer. "
            "We need to adjust the phone mapping to this model's vocabulary."
        )

    return candidates


# ------------------------- FILLER / HESITATION FILTER -------------------------

FILLER_WORDS = ("um", "umm", "uh", "uhh", "erm", "er", "hmm", "mm")


def build_filler_phone_sequences(tokenizer):
    """Build model-token patterns for common fillers using CMUdict."""
    patterns = []
    dictionary = cmudict.dict()
    for filler in FILLER_WORDS:
        for arpabet in dictionary.get(filler, []):
            try:
                symbols = arpabet_to_model_symbols(arpabet)
                encoded = tokenizer(" ".join(symbols), add_special_tokens=False,
                                    return_attention_mask=False)
                ids = list(encoded["input_ids"])
                if not ids or (tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids):
                    continue
                phones = tokenizer.convert_ids_to_tokens(ids)
                if phones:
                    patterns.append((filler, phones))
            except Exception:
                continue
    unique = {}
    for name, phones in patterns:
        unique[tuple(phones)] = name
    return sorted([(name, list(seq)) for seq, name in unique.items()],
                  key=lambda item: len(item[1]), reverse=True)


def remove_edge_fillers(actual_phones, filler_sequences):
    """Remove recognised filler sequences only at the beginning or end."""
    cleaned = list(actual_phones)
    removed = []
    changed = True
    while changed and cleaned:
        changed = False
        for name, sequence in filler_sequences:
            if cleaned[:len(sequence)] == sequence:
                del cleaned[:len(sequence)]
                removed.append(name)
                changed = True
                break
        if changed:
            continue
        for name, sequence in filler_sequences:
            if len(sequence) <= len(cleaned) and cleaned[-len(sequence):] == sequence:
                del cleaned[-len(sequence):]
                removed.append(name)
                changed = True
                break
    return cleaned, removed


# ------------------------- RECORDING -------------------------



def record_one_attempt():

    """Record one 16 kHz mono utterance and stop after trailing silence."""

    frames = []

    calibration = []

    pre_roll = []



    calibration_blocks = max(

        1, round(CALIBRATION_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    )

    pre_roll_limit = max(

        1, round(PRE_ROLL_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    )



    speech_started = False

    silent_seconds = 0.0

    session_start = time.monotonic()

    speech_start = None



    print("\nListening... Say the word after the short pause.")



    try:

        with sd.InputStream(

            samplerate=SAMPLE_RATE,

            channels=CHANNELS,

            dtype="int16",

            blocksize=BLOCK_SIZE,

        ) as stream:

            while True:

                data, overflowed = stream.read(BLOCK_SIZE)

                if overflowed:

                    print("\nWarning: microphone buffer overflow.")



                chunk = np.asarray(data[:, 0], dtype=np.int16).copy()

                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

                elapsed = time.monotonic() - session_start



                if len(calibration) < calibration_blocks:

                    calibration.append(rms)

                    pre_roll.append(chunk)

                    if len(pre_roll) > pre_roll_limit:

                        pre_roll.pop(0)

                    continue



                noise_floor = float(np.median(calibration))

                threshold = max(MIN_SPEECH_RMS, noise_floor * NOISE_MULTIPLIER)



                if rms >= threshold:

                    if not speech_started:

                        speech_started = True

                        speech_start = time.monotonic()

                        frames.extend(pre_roll)

                        print("Speech detected.")

                    frames.append(chunk)

                    silent_seconds = 0.0



                elif speech_started:

                    frames.append(chunk)

                    silent_seconds += len(chunk) / SAMPLE_RATE

                    if silent_seconds >= SILENCE_SECONDS:

                        print("Recording finished.")

                        break



                else:

                    if elapsed >= START_TIMEOUT_SECONDS:

                        print("No speech detected. Please try again.")

                        return None

                    pre_roll.append(chunk)

                    if len(pre_roll) > pre_roll_limit:

                        pre_roll.pop(0)



                if (

                    speech_start is not None

                    and time.monotonic() - speech_start >= MAX_RECORD_SECONDS

                ):

                    print("Maximum recording time reached.")

                    break



    except Exception as exc:

        print(f"Microphone error: {type(exc).__name__}: {exc}")

        return None



    if not frames:

        return None



    audio_int16 = np.concatenate(frames)

    audio = audio_int16.astype(np.float32) / 32768.0



    if len(audio) / SAMPLE_RATE < MIN_AUDIO_SECONDS:

        print("Recording was too short. Please try again.")

        return None



    peak = float(np.max(np.abs(audio)))

    rms = float(np.sqrt(np.mean(audio ** 2)))

    print(f"Captured {len(audio) / SAMPLE_RATE:.2f} seconds.")

    print(f"Audio level — peak: {peak:.3f}, RMS: {rms:.4f}")



    if peak < 0.005:

        print("The recording is very quiet. Check microphone volume and distance.")



    return audio





# ------------------------- PHONEME DETECTION -------------------------



def detect_phonemes(audio, model, tokenizer, feature_extractor, device):

    if audio is None or len(audio) == 0:

        return []



    inputs = feature_extractor(

        audio,

        sampling_rate=SAMPLE_RATE,

        return_tensors="pt",

        padding=True,

    )

    input_values = inputs.input_values.to(device)



    with torch.inference_mode():

        logits = model(input_values).logits



    # Standard CTC greedy decoding, as shown in the model card.

    predicted_ids = torch.argmax(logits, dim=-1)

    decoded = tokenizer.batch_decode(predicted_ids, group_tokens=True)[0]



    # The phoneme tokenizer separates phones with spaces.

    return [p for p in decoded.strip().split() if p]





# ------------------------- ALIGNMENT / SCORE -------------------------



def align_phones(expected, actual):

    """Levenshtein alignment with match, substitution, omission and addition."""

    n, m = len(expected), len(actual)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    back = [[None] * (m + 1) for _ in range(n + 1)]



    for i in range(1, n + 1):

        dp[i][0] = i

        back[i][0] = "omission"



    for j in range(1, m + 1):

        dp[0][j] = j

        back[0][j] = "addition"



    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if expected[i - 1] == actual[j - 1]:

                diagonal = (dp[i - 1][j - 1], "match")

            else:

                diagonal = (dp[i - 1][j - 1] + 1, "substitution")



            choices = [

                diagonal,

                (dp[i - 1][j] + 1, "omission"),

                (dp[i][j - 1] + 1, "addition"),

            ]

            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])



    operations = []

    i, j = n, m

    while i > 0 or j > 0:

        op = back[i][j]

        if op in ("match", "substitution"):

            operations.append({

                "type": op,

                "expected": expected[i - 1],

                "actual": actual[j - 1],

            })

            i -= 1

            j -= 1

        elif op == "omission":

            operations.append({

                "type": op,

                "expected": expected[i - 1],

                "actual": None,

            })

            i -= 1

        elif op == "addition":

            operations.append({

                "type": op,

                "expected": None,

                "actual": actual[j - 1],

            })

            j -= 1

        else:

            break



    operations.reverse()

    return dp[n][m], operations





def assess(expected, actual):

    if not expected or not actual:

        return {"score": None, "status": "UNABLE_TO_ASSESS", "operations": []}



    # A very long/unexpected decode is more likely to be unreliable than a fair

    # one-word comparison. Do not turn it into a misleading low score.

    if len(actual) > len(expected) + MAX_EXTRA_PHONES:

        return {"score": None, "status": "UNABLE_TO_ASSESS", "operations": []}



    distance, operations = align_phones(expected, actual)

    denominator = max(len(expected), len(actual), 1)

    score = round(max(0.0, (1.0 - distance / denominator) * 100), 1)



    return {

        "score": score,

        "status": "LIKELY_MATCH" if distance == 0 else "POSSIBLE_DIFFERENCE",

        "operations": operations,

    }





# ------------------------- CHILD-FRIENDLY RESULT -------------------------



def make_feedback(word, result):

    if result["score"] is None:

        return "I couldn't hear that clearly. Let's try one more time."



    errors = [op for op in result["operations"] if op["type"] != "match"]

    if not errors:

        return f"Great effort! The sounds in {word} matched the practice word."



    first = errors[0]

    if first["type"] == "substitution":

        return (

            f"Good try! Let's practise the /{first['expected']}/ sound "

            f"in {word} together."

        )

    if first["type"] == "omission":

        return (

            f"Good try! Let's practise adding the /{first['expected']}/ "

            f"sound in {word}."

        )

    return (

        f"Good try! Let's practise saying {word} without the extra "

        f"/{first['actual']}/ sound."

    )





def print_result(word, expected_ipa, expected, actual, result):

    print("\n" + "=" * 44)

    print("          voAIce SPEECH BUDDY")

    print("=" * 44)

    print(f"Target word: {word}")



    print("\nEXPECTED SOUNDS:")

    print("  " + "  ".join(f"/{p}/" for p in expected))



    print("\nSOUNDS I HEARD:")

    if actual:

        print("  " + "  ".join(f"/{p}/" for p in actual))

    else:

        print("  [UNCLEAR]")



    print("\nSOUND CHECK:")

    if result["score"] is None:

        print("  Unable to compare the sounds.")

    else:

        for op in result["operations"]:

            if op["type"] == "match":

                print(f"  /{op['expected']}/  ->  /{op['actual']}/ : MATCH")

            elif op["type"] == "substitution":

                print(

                    f"  /{op['expected']}/  ->  /{op['actual']}/ : "

                    "POSSIBLE SUBSTITUTION"

                )

            elif op["type"] == "omission":

                print(f"  /{op['expected']}/ : POSSIBLE OMISSION")

            elif op["type"] == "addition":

                print(f"  /{op['actual']}/ : POSSIBLE ADDITION")



    if result["score"] is None:

        print("\nPRACTICE SCORE: Unable to assess")

    else:

        print(f"\nPRACTICE SCORE: {result['score']:.1f}%")



    print("\nFEEDBACK:")

    print("  " + make_feedback(word, result))

    print(f"\nSTATUS: {result['status']}")

    print("=" * 44)





# ------------------------- PRACTICE SESSION -------------------------



def choose_level(words):

    levels = sorted({item["level"] for item in words})

    print("\nAvailable levels:")

    for level in levels:

        print(f"  {level}. Level {level}")



    while True:

        choice = input("\nChoose level (or q to quit): ").strip().lower()

        if choice == "q":

            return None

        if choice.isdigit() and int(choice) in levels:

            return int(choice)

        print("Please enter one of the listed levels.")





def practise_level(words, level, model, tokenizer, extractor, device, filler_sequences):

    level_words = [item for item in words if item["level"] == level]

    random.shuffle(level_words)



    print(f"\nStarting Level {level}.")

    print("Press Enter when ready for each word. Type q to stop.")

    print("Hesitation sounds such as um, uh and erm are filtered at the start/end.")
    print("TTS is not included; your teammate can connect it separately.")



    session_results = []



    for index, item in enumerate(level_words, start=1):

        word = item["target_word"]

        command = input(

            f"\nWord {index}/{len(level_words)} — {word} (Enter/q): "

        ).strip().lower()



        if command == "q":

            break



        try:

            expected_candidates = make_expected_pronunciations(word, tokenizer)

        except Exception as exc:

            print(f"\nExpected pronunciation error for {word!r}:")

            print(f"  {exc}")

            print("The word was skipped. Check spelling, CMUdict, and the phone mapping.")

            continue



        print(f"Found {len(expected_candidates)} CMUdict pronunciation(s).")



        final_result = None

        for attempt in range(1, 3):

            print(f"\nAttempt {attempt} of 2")

            audio = record_one_attempt()



            if audio is None:

                actual_phones = []

            else:

                try:

                    actual_phones = detect_phonemes(

                        audio, model, tokenizer, extractor, device

                    )

                except Exception as exc:

                    print(f"Model decoding error: {type(exc).__name__}: {exc}")

                    actual_phones = []



            actual_phones, ignored_fillers = remove_edge_fillers(
                actual_phones, filler_sequences
            )
            if ignored_fillers:
                print("Ignored hesitation sound(s): " + ", ".join(ignored_fillers))

            comparisons = [
                (candidate, assess(candidate["phones"], actual_phones))
                for candidate in expected_candidates
            ]
            usable = [item for item in comparisons if item[1]["score"] is not None]
            if usable:
                selected_candidate, result = max(usable, key=lambda item: item[1]["score"])
            else:
                selected_candidate = expected_candidates[0]
                result = assess(selected_candidate["phones"], actual_phones)
            expected_ipa = selected_candidate["ipa"]
            expected_phones = selected_candidate["phones"]
            print(f"CMUdict ARPAbet: {selected_candidate['arpabet']}")
            print_result(word, expected_ipa, expected_phones, actual_phones, result)



            final_result = {

                "word": word,

                "score": result["score"],

                "status": result["status"],

                "attempts": attempt,

            }



            if result["status"] == "LIKELY_MATCH":

                break



            if attempt == 1:

                input("\nPress Enter when you are ready to try again...")



        if final_result:

            session_results.append(final_result)



    print("\n" + "=" * 44)

    print(f"LEVEL {level} SESSION SUMMARY")

    print("=" * 44)



    if not session_results:

        print("No words were assessed.")

        return



    assessed = [r for r in session_results if r["score"] is not None]

    print(f"Words completed: {len(session_results)}")

    print(f"Unable to assess: {len(session_results) - len(assessed)}")



    if assessed:

        average = float(np.mean([r["score"] for r in assessed]))

        print(f"Average practice score: {average:.1f}%")

    else:

        print("Average practice score: unavailable")



    print("\nIndividual results:")

    for i, item in enumerate(session_results, start=1):

        score = f"{item['score']:.1f}%" if item["score"] is not None else "N/A"

        print(

            f"{i}. {item['word']} | {score} | {item['status']} "

            f"| Attempts: {item['attempts']}"

        )



    print("\nSession completed. Audio was not saved.")





# ------------------------- STARTUP CHECK -------------------------


def check_cmudict():
    """Check that CMUdict is installed and contains the sample words."""
    try:
        entries = cmudict.dict()
        for sample in ("rabbit", "apple"):
            if sample not in entries:
                raise RuntimeError(f"CMUdict test word '{sample}' was not found.")
        print("CMU Pronouncing Dictionary is ready.")
        print("Example: rabbit -> " + " ".join(entries["rabbit"][0]))
        return True
    except Exception as exc:
        print("\nERROR: CMUdict is not ready.")
        print(str(exc))
        print("Install it with: python -m pip install cmudict")
        return False


# ------------------------- MAIN -------------------------



def main():

    if not check_cmudict():

        return



    try:

        words = load_words()

    except Exception as exc:

        print(f"Could not load word list: {exc}")

        return



    try:

        model, tokenizer, extractor, device = load_model()

    except Exception as exc:

        print(f"Could not load model: {type(exc).__name__}: {exc}")

        print("Check internet access and your installed package versions.")

        return



    filler_sequences = build_filler_phone_sequences(tokenizer)
    print(f"Filler filter ready ({len(filler_sequences)} recognised patterns).")

    print("\nvoAIce Speech Buddy — local phoneme practice")

    print(f"Loaded {len(words)} target words.")

    print("Audio is processed locally and is not saved or uploaded.")



    try:

        while True:

            level = choose_level(words)

            if level is None:

                print("Goodbye!")

                break

            practise_level(

                words, level, model, tokenizer, extractor, device, filler_sequences

            )

    except KeyboardInterrupt:

        print("\nPractice stopped.")





if __name__ == "__main__":

    main()
