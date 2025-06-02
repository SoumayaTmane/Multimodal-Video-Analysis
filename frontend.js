import React, { useState } from 'react';
import './App.css'; 

function App() {
  const [youtubeLink, setYoutubeLink] = useState('');
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [videoId, setVideoId] = useState(''); // To store the extracted video ID

  // Function to extract video ID (simplified for frontend)
  const getYouTubeVideoId = (url) => {
    try {
      const parsedUrl = new URL(url);
      if (parsedUrl.hostname.includes('youtube.com')) {
        const queryParams = new URLSearchParams(parsedUrl.search);
        return queryParams.get('v');
      } else if (parsedUrl.hostname.includes('youtu.be')) {
        return parsedUrl.pathname.slice(1);
      }
      return null;
    } catch (e) {
      console.error("Invalid URL:", e);
      return null;
    }
  };

const processVideoAndChat = async () => {
    setError('');
    setIsLoading(true);
    setMessages([]);

    const id = getYouTubeVideoId(youtubeLink);
    if (!id) {
      setError("Please enter a valid YouTube URL.");
      setIsLoading(false);
      return;
    }
    setVideoId(id);

    try {
      const response = await axios.post('http://localhost:5000/process_video', { // Flask server runs on port 5000 by default
        video_url: youtubeLink
      }, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded', // Important for Flask
        },
      });

      const data = response.data;

      if (data.error) {
        setError(data.error);
      } else {
        setMessages(prev => [...prev, { sender: 'Gemini', text: data.message }]);
        setMessages(prev => [...prev, { sender: 'Gemini', text: `**Video Summary:**\n${data.summary}` }]);
        setMessages(prev => [...prev, { sender: 'Gemini', text: `**Video Sections:**\n${data.sections}` }]); // Assuming sections are already in Markdown
      }
    } catch (err) {
      setError("Failed to process video. Please check the URL and your backend connection.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };


const handleSendMessage = async () => {
    if (currentMessage.trim() === '') return;

    const userMessage = { sender: 'User', text: currentMessage };
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');

    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/chat', { // Flask server runs on port 5000 by default
        user_query: currentMessage,
        video_id: videoId
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = response.data;
      setMessages(prev => [...prev, { sender: 'Gemini', text: data.response }]);
    } catch (err) {
      setError("Failed to get response from Gemini. Please try again.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      {/* ... (rest of your React component) ... */}
    </div>
  );
}

export default App;