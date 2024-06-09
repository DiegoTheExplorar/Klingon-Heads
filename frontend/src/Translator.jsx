import { Client } from "@gradio/client";
import React, { useState } from 'react';
import './Translator.css'; // Import the CSS file

function Translator() {
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

  return (
    <div className="container mx-auto px-4">
      <header className="text-center my-12">
        <img src="/image.png" alt="Klingon Heads Logo" className="logo mx-auto" />
      </header>
      <div className="flex justify-between items-center space-x-4">
        <div className="flex-1">
          <textarea 
            className="w-full p-4 h-60 bg-gray-100 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Enter ${input ? "English" : "Klingon"} text`}
          />
        </div>
        <button 
          className="p-2 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded transform rotate-90"
          onClick={toggleTranslationDirection}>
          &#8596;
        </button>
        <div className="flex-1">
          <textarea 
            className="w-full p-4 h-60 bg-gray-200 rounded border border-gray-300 focus:outline-none"
            value={translation}
            readOnly
          />
        </div>
      </div>
      <div className="text-center my-8">
        <button 
          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
          onClick={translateText}>
          Translate
        </button>
        <button 
          className="ml-4 bg-yellow-500 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded"
          onClick={handleFavourite}>
          Add to Favourites
        </button>
      </div>
    </div>
  );
}

export default Translator;
