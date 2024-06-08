import { Client } from "@gradio/client";
import React, { useState } from 'react';
import './App.css'; // Import the CSS file

function App() {
  const [input, setInput] = useState('');
  const [translation, setTranslation] = useState('');
  const [history, setHistory] = useState([]); // State to keep track of history
  const [favourites, setFavourites] = useState([]); // State to keep track of favourites
  const [translateToKlingon, setTranslateToKlingon] = useState(true); // State to toggle translation direction

  const translateText = async () => {
    if (!input.trim()) {
      alert(`Please enter some ${translateToKlingon ? "English" : "Klingon"} text.`);
      return;
    }

    try {
      const client = await Client.connect("DiegoTheExplorar/KlingonHeads");
      const apiEndpoint = translateToKlingon ? "/predict" : "/reverse_predict"; // Assuming /reverse_predict is the endpoint for Klingon to English
      const data = translateToKlingon ? { english_sentence: input } : { klingon_sentence: input };
      const result = await client.predict(apiEndpoint, data);
      setTranslation(result.data);
      setHistory(prev => [...prev, { input, translation: result.data }]);
    } catch (error) {
      console.error('Failed to translate:', error);
      setTranslation('Error: Failed to translate');
    }
  };

  const handleFavourite = () => {
    if (!translation) return;
    setFavourites(prev => [...prev, { input, translation }]);
  };

  const toggleTranslationDirection = () => {
    setTranslateToKlingon(!translateToKlingon);
    setInput(translation); // Swap input and translation fields
    setTranslation('');
  };

  return (
    <div className="container">
      <header>
        <img src="/image.png" alt="Klingon Heads Logo" className="logo" />
      </header>
      <div className="translation-container">
        <div className="translation-box">
          <label>{translateToKlingon ? "English" : "Klingon"}</label>
          <textarea 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Enter ${translateToKlingon ? "English" : "Klingon"} text`}
          />
        </div>
        <div className="arrows" onClick={toggleTranslationDirection}>&#8596;</div>
        <div className="translation-box">
          <label>{translateToKlingon ? "Klingon" : "English"}</label>
          <textarea 
            value={translation}
            readOnly
          />
        </div>
      </div>
      <div className="buttons-container">
        <button onClick={translateText}>Translate</button>
        <button onClick={handleFavourite} className="favourite-button">Add to Favourites</button>
        <div className="icons">
          <div className="icon" onClick={() => alert('Showing history...')}>History</div>
          <div className="icon" onClick={() => alert('Showing favourites...')}>Favourites</div>
        </div>
      </div>
    </div>
  );
}

export default App;
