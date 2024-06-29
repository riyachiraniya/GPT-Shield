# Plagiarazzi: AI Plagiarism Detector

Plagiarazzi is a tool designed to detect AI-generated content using GPT-2. It analyzes the input text and provides insights based on perplexity and burstiness scores. The tool is built with Streamlit for an interactive web interface.

## Features

- **Perplexity Score Calculation**: Measures how well a language model predicts a sample, with lower scores indicating better predictions.
- **Burstiness Score Calculation**: Analyzes the frequency of repeated words in the text.
- **Top Repeated Words Visualization**: Displays a bar chart of the top 10 most repeated words in the input text.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- `pip` (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/riyachiraniya/plagiarazzi.git
    cd plagiarazzi
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the Streamlit application:

    ```bash
    streamlit run plagiarazzi.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Enter the text you want to analyze in the text area and click "Analyze".

### Output

- **Your Input Text**: Displays the entered text.
- **Calculated Score**: Shows the perplexity and burstiness scores.
- **Basic Insights**: Provides a bar chart of the top 10 most repeated words.

## Disclaimer

Plagiarazzi provides an estimation of originality but is not a foolproof method. Human review and verification are necessary to confirm results. By using it, you acknowledge that you have read and understood these disclaimers and agree to use the tool responsibly.


## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [NLTK](https://www.nltk.org/)
- [Plotly](https://plotly.com/)

## Contact

For questions or comments, please open an issue on GitHub.

---
