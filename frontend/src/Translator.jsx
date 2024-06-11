import { Client } from "@gradio/client";
import accountIcon from '@iconify-icons/mdi/account';
import heartIcon from '@iconify-icons/mdi/heart';
import historyIcon from '@iconify-icons/mdi/history';
import microphoneIcon from '@iconify-icons/mdi/microphone';
import { Icon } from '@iconify/react';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Translator.css';
import { addFavoriteToFirestore, addHistoryToFirestore } from "./firebasehelper"; // Import function to add favorites and history to Firestore

function Translator() {
  const [input, setInput] = useState('');
  const [translation, setTranslation] = useState('');
  const [translateToKlingon, setTranslateToKlingon] = useState(true); // State to toggle translation direction
  const [isFavourite, setIsFavourite] = useState(false); // State for favourite button
  const [showDropdown, setShowDropdown] = useState(false); // State for user icon dropdown
  const [isListening, setIsListening] = useState(false);
  const [speechRecognition, setSpeechRecognition] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onresult = event => {
        setInput(event.results[0][0].transcript);
      };
      setSpeechRecognition(recognition);
    } else {
      alert('Speech recognition not available. Please use Chrome to use this feature');
    }
  }, []);

  const toggleListening = () => {
    if (!isListening) {
      speechRecognition.start();
    } else {
      speechRecognition.stop();
    }
    setIsListening(!isListening);
  };


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
      const ETK = await Client.connect("DiegoTheExplorar/KlingonHeads");
      const KTE = await Client.connect("DiegoTheExplorar/KlingonToEnglish");
      const client = translateToKlingon ? ETK : KTE;
      const data = translateToKlingon ? { english_sentence: input } : { klingon_sentence: input };
      const result = await client.predict("/predict", data);
      setTranslation(result.data);
      addHistoryToFirestore(input, result.data);
    } catch (error) {
      console.error('Failed to translate:', error);
      setTranslation('Error: Failed to translate');
    }
  };

  const handleFavourite = async () => {
    if (!translation) return;

    try {
      await addFavoriteToFirestore(input, translation); // Call function to add favorite to Firestore
      setIsFavorite(true);
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
    setIsFavourite(false); 
  };

  return (
    <div className="container">
      <header className="text-center my-4">
        <img src="/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" className="logo" />
      </header>
      <div className="user-icon-container" onClick={() => setShowDropdown(!showDropdown)}>
        <Icon icon={accountIcon} className="user-icon" />
        {showDropdown && (
          <div className="dropdown-menu">
            <button onClick={handleSignOut}>Sign Out</button>
            <button onClick={() => navigate('/profile')}>Profile</button>
          </div>
        )}
      </div>
      <div className="translation-container">
        <div className="english-input-container">
          <label htmlFor="english">{translateToKlingon ? "English" : "Klingon"}</label>
          <textarea
            id="english"
            className="input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Enter ${translateToKlingon ? "English" : "Klingon"} text`}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                translateText();
              }
            }}
          />
          <button onClick={toggleListening} className="mic-button">
            <Icon icon={microphoneIcon} className="mic-icon" style={{ color: isListening ? 'red' : 'black' }} />
          </button>
        </div>
        <button className="swap-button" onClick={toggleTranslationDirection}>
          ↔
        </button>
        <div className="klingon-output-container">
          <label htmlFor="klingon">{translateToKlingon ? "Klingon" : "English"}</label>
          <textarea
            id="klingon"
            className="input"
            value={translation}
            readOnly
          />
          <button className="fav-button" onClick={handleFavourite}>
            <Icon icon={isFavourite ? heartFilledIcon : heartIcon} className="fav-icon" />
          </button>
        </div>
      </div>
      <div className="footer">
        <div className="footer-icon-container" onClick={showHistory}>
          <Icon icon={historyIcon} className="footer-icon" />
          <span>History</span>
        </div>
        <div className="footer-icon-container" onClick={showFav}>
          <Icon icon={heartIcon} className="footer-icon" />
          <span>Favourites</span>
        </div>
      </div>
    </div>
  );
}



export default Translator;
