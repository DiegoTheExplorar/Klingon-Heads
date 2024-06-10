import { Client } from "@gradio/client";
import React, { useState } from 'react';
import HistoryPage from './HistoryPage'; // Import the HistoryPage component
import './Translator.css'; // Import the CSS file

function Translator() {
  const [input, setInput] = useState('');
  const [translation, setTranslation] = useState('');
  const [history, setHistory] = useState([]); // State to keep track of history
  const [favourites, setFavourites] = useState([]); // State to keep track of favourites
  const [translateToKlingon, setTranslateToKlingon] = useState(true); // State to toggle translation direction
  const [showHistory, setShowHistory] = useState(false); // State to control showing the history

  const translateText = async () => {
    if (!input.trim()) {
      alert(`Please enter some ${translateToKlingon ? "English" : "Klingon"} text.`);
      return;
    }

    try {
      const client = await Client.connect("DiegoTheExplorar/KlingonHeads");
      const apiEndpoint = translateToKlingon ? "/predict" : "/reverse_predict"; 
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

  const toggleHistoryVisibility = () => {
    setShowHistory(!showHistory);
  };

  const clearTextAreas = () => {
    setInput('');
    setTranslation('');
  };

  return (
    <div className="container">
      <header className="text-center my-4">
        <img src="/image.png" alt="Klingon Heads Logo" className="logo" />
      </header>
      <div className="flex justify-between items-center space-x-4">
        <div className="flex-1">
          <textarea 
            className="input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Enter ${translateToKlingon ? "English" : "Klingon"} text`}
          />
        </div>
        <button 
          className="button"
          onClick={toggleTranslationDirection}>
          &#8596;
        </button>
        <div className="flex-1">
          <textarea 
            className="input"
            value={translation}
            readOnly
          />
        </div>
      </div>
      <div className="text-center my-4">
        <button 
          className="translate-button"
          onClick={translateText}>
          Translate
        </button>
        <button 
          className="favourite-button"
          onClick={handleFavourite}>
          Add to Favourites
        </button>
        <button 
          className="history-button"
          onClick={toggleHistoryVisibility}>
          Toggle History
        </button>
        <button 
          className="clear-button"
          onClick={clearTextAreas}>
          Clear
        </button>
      </div>
      {showHistory && <HistoryPage history={history} />}
    </div>
  );
}

export default Translator;
