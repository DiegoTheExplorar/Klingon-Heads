import { Client } from "@gradio/client";
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Translator.css';
import { addFavoriteToFirestore, addHistoryToFirestore } from "./firebasehelper"; // Import function to add favorites and history to Firestore

function Translator() {
  const [input, setInput] = useState('');
  const [translation, setTranslation] = useState('');
  const [translateToKlingon, setTranslateToKlingon] = useState(true); // State to toggle translation direction
  const navigate = useNavigate();

  const handleSignOut = () => {
    // Clear stored user info like email or auth tokens
    localStorage.removeItem('email'); // Assuming email is stored in localStorage
    navigate('/'); // Navigate back to sign-in page
  };

  const showFav = () => {
    navigate('/fav'); // Navigate to favourites
  };

  const showHistory = () => {
    navigate('/history'); // Navigate to history
  };

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
      addHistoryToFirestore(input, result.data); // Call function to add translation history to Firestore
    } catch (error) {
      console.error('Failed to translate:', error);
      setTranslation('Error: Failed to translate');
    }
  };

  const handleFavourite = async () => {
    if (!translation) return;

    try {
      await addFavoriteToFirestore(input, translation); // Call function to add favorite to Firestore
      alert('Added to favourites!');
    } catch (error) {
      console.error("Error adding document: ", error);
      alert('Failed to add to favourites.');
    }
  };

  const toggleTranslationDirection = () => {
    setTranslateToKlingon(!translateToKlingon);
    setInput(translation); // Swap input and translation fields
    setTranslation('');
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
          onClick={showHistory}>
          See your past translations
        </button>
        <button 
          className="clear-button"
          onClick={clearTextAreas}>
          Clear
        </button>
        <button 
          className="sign-out-button"
          onClick={handleSignOut}>
          Sign Out
        </button>
        <button 
          className="show-fav"
          onClick={showFav}>
          See your favorites
        </button>
      </div>
    </div>
  );
}

export default Translator;
