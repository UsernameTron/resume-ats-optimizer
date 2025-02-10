# Resume ATS Optimizer

An AI-powered tool that helps optimize resumes for Applicant Tracking Systems (ATS) by analyzing keyword matches, providing detailed recommendations, and suggesting improvements based on job descriptions.

## Features

- **ATS Match Score**: Calculate how well your resume matches the job description
- **Keyword Analysis**: Identify missing important keywords from the job description
- **Smart Recommendations**: Get AI-powered suggestions to improve your resume
- **Word-Friendly Output**: Copy-paste friendly formatting for easy editing
- **Downloadable Reports**: Save optimization reports as text files

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/resume-ats-optimizer.git
   cd resume-ats-optimizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your-api-key-here`

## Usage

1. Start the application:
   ```bash
   streamlit run keyword_density_ats.py
   ```

2. In the web interface:
   - Paste your resume in the left text area
   - Paste the job description in the right text area
   - Click "Analyze and Optimize Resume"

3. Review the results:
   - Check your ATS match score
   - Review missing keywords
   - Read the detailed recommendations
   - Copy the suggestions or download the report

## Recent Updates

- Improved Word-friendly text formatting
- Enhanced Streamlit interface with better layout
- Added download option for reports
- Fixed special character handling
- Added loading spinner during analysis
- Improved copy-paste functionality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
