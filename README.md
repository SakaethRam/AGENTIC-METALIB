# MetaLib GPT

MetaLib GPT is an interactive AI chat assistant powered by the Gemini API, built with React. It simulates a conversation with a GPT-style AI and provides rich formatting, context persistence, and user-friendly interactions. It is designed as part of the MetaLib ecosystem to help users explore ideas, solve problems, and engage with AI effectively.

---

![README_HOME](https://github.com/user-attachments/assets/b8e7adb5-88b5-4a1f-94ef-34bc536a1074)

![README_DESCRIPTION](https://github.com/user-attachments/assets/1a6f35f1-e11e-49ef-8f9d-cd4405570292)

---
## Features

- Context Management: Maintains previous prompts and recent interactions using React Context API.
- Interactive UI: Includes a dynamic sidebar, prompt suggestion cards, and chat interface.
- Prompt Suggestions: Clickable cards to instantly trigger predefined prompts.
- Response Formatting: Renders bold, line breaks, and code blocks from raw text.
- Loading Indicator: Animated loader during API processing.
- Chat History Access: Sidebar shows recent prompts to easily reload previous queries.
- Built on Gemini API: Processes and returns AI-generated responses from your configured model.

---

## Technology Stack

- Frontend: React
- State Management: React Context API
- Styling: CSS
- AI Integration: Gemini API (`run()` from `../Config/Gemini`)

---

## Folder Structure

```
/MetaLib-GPT
│
├── /assets/             # Icons and images
├── /Context/
│   └── Context.jsx      # Context Provider for global state
├── /Components/
│   ├── Sidebar.jsx      # Sidebar with recent chats & options
│   └── Main.jsx         # Main UI component
├── /Config/
│   └── Gemini.js        # API integration with Gemini (not shown)
├── App.js
└── index.js
```

---

## Running Locally

1. Clone the repository:
   ```bash
   git clone "https://github.com/SakaethRam/MetaLib.git"
   cd MetaLib-GPT
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure Gemini API:
   - Ensure your `run()` function in `Config/Gemini.js` is set up to call the Gemini model.

4. Run the application:
   ```bash
   npm start
   ```

---

## Sample Prompts

- "Suggest beautiful places to see on an upcoming road trip"
- "Briefly summarize this concept: Urban planning"
- "Brainstorm team bonding activities for our work retreat"
- "Tell me about React js and React native"

---

## Disclaimer

MetaLib GPT may display inaccurate information, especially regarding people or sensitive topics. Always verify its responses. All interactions remain local unless explicitly configured to use external services.

---

## Future Enhancements

- Voice input/output integration
- Multi-language support
- Role-based personality customization
- Chat export and history synchronization

---

## Docker Support

### Dockerfile

```Dockerfile
# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the React app
RUN npm run build

# Serve the build with a simple static server
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]
```

### Build and Run

1. Build the Docker image:
```bash
docker build -t metalib-gpt .
```

2. Run the container:
```bash
docker run -p 3000:3000 metalib-gpt
```

The application will be accessible at `http://localhost:3000`.

---

## Powered By

> [MetaLib GPT](https://zrex.netlify.app)

---

Built by `S A M` – Bringing intelligence to text.
