# Medical Slot Extraction with LLM

This Python script uses an LLM (via Ollama API) to extract structured medical information from unstructured medical context text. It asynchronously queries the model for multiple medical "slots" (e.g., medical history, chief complaints) and outputs JSON-formatted summaries.

## Features

- Async calls to Ollama LLM API to generate structured JSON for predefined medical slots.
- Slot-specific JSON schema enforcement.
- Robust JSON parsing with error handling.
- Color-coded CLI output for easier reading.
- Fully async with concurrency for faster completion.

## Requirements

- Python 3.7+
- [Ollama API](https://ollama.com/) running locally at `http://localhost:11434`
- Packages listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Ollama API locally with the appropriate model (e.g., `llama3.2:3b`).
2. Run the script:

```bash
python your_script_name.py
```

3. Paste the unstructured medical text context and submit by pressing Enter on an empty line.
4. Wait for the script to asynchronously generate JSON outputs for each medical slot.

## Configuration

- Change the `OLLAMA_URL` and `MODEL` constants at the top of the script if needed.

## Example Slots

- Brief Medical History
- Chief Complaints
- Current Symptoms and Medical Background
- Past Medical History
- Hospitalization and Surgical History
- Gynecological History
- Lifestyle and Social Activity
- Family History
- Allergies and Hypersensitivities

## License

See [LICENSE.md](LICENSE.md).

---

*Feel free to open issues or contribute!*
