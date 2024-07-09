import { Client } from "@gradio/client";
import closeIcon from '@iconify-icons/ic/twotone-close';
import translateIcon from '@iconify-icons/mdi/arrow-forward';
import cameraIcon from '@iconify-icons/mdi/camera';
import copyIcon from '@iconify-icons/mdi/content-copy';
import heartIcon from '@iconify-icons/mdi/heart';
import historyIcon from '@iconify-icons/mdi/history';
import microphoneIcon from '@iconify-icons/mdi/microphone';
import swapIcon from '@iconify-icons/mdi/swap-horizontal-bold';
import speakerIcon from '@iconify-icons/mdi/volume-high';
import { Icon } from '@iconify/react';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { useNavigate } from 'react-router-dom';
import Tesseract from 'tesseract.js';
import { addFavoriteToFirestore, addHistoryToFirestore, checkFavoriteInFirestore, removeFavoriteBasedOnInput } from './FireBase/firebasehelper';
import './Translator.css';
import UserDropdown from './UserDropdown';

function Translator() {
  const [input, setInput] = useState('');
  const [translation, setTranslation] = useState('');
  const [translating, setTranslating] = useState(false);
  const [translateToKlingon, setTranslateToKlingon] = useState(true); // State to toggle translation direction
  const [isFavourite, setIsFavourite] = useState(false); // State for favourite button
  const [showDropdown, setShowDropdown] = useState(false); // State for user icon dropdown
  const [isListening, setIsListening] = useState(false);
  const [speechRecognition, setSpeechRecognition] = useState(null);
  const [profilePicUrl, setProfilePicUrl] = useState(null);
  const navigate = useNavigate();
  const auth = getAuth();


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

  
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      if (user) {
        // User is signed in
        setProfilePicUrl(user.photoURL);
      } else {
        // No user is signed in
        setProfilePicUrl(null);
        console.log("No user is signed in.");
      }
    });
  
  
    return () => unsubscribe();
  }, []); 
  


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
      setTranslating(true);
      const ETK = await Client.connect("DiegoTheExplorar/KlingonHeads");
      const KTE = await Client.connect("DiegoTheExplorar/KlingonHeads_Klingon_To_English");
      const client = translateToKlingon ? ETK : KTE;
      const data = translateToKlingon ? { english_sentence: input } : { klingon_sentence: input };
      const result = await client.predict("/predict", data);
      setTranslation(result.data);
      addHistoryToFirestore(input, result.data,translateToKlingon ? "English" : "Klingon");
      FavinDB();
    } catch (error) {
      console.error('Failed to translate:', error);
      setTranslation('Error: Failed to translate');
    } finally {
      setTranslating(false);
    }
  };

  const FavinDB = async() => {
    if(!input) return;

    try{
      const exists = await checkFavoriteInFirestore(input);
      console.log("FavinDB checked: ", exists);
      setIsFavourite(exists);
      return exists;
    } catch (error){
      console.error('Error checking favorites:', error);
      return false;
    }
    
  }

  const handleFavourite = async () => {
    if (!translation) return;

    try {
      await addFavoriteToFirestore(input, translation,translateToKlingon ? "English" : "Klingon"); 
      setIsFavourite(true);
      //alert('Added to favourites!');
    } catch (error) {
      console.error("Error adding document: ", error);
      alert('Failed to add to favourites.');
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Tesseract.recognize(
        file,
        'eng', 
        {
          logger: (m) => console.log(m),
        }
      )
        .then(({ data: { text } }) => {
          console.log("Recognized Text:", text);
          setInput(text);
        })
        .catch(error => {
          console.error('Failed to recognize text:', error);
          alert('Error: Failed to recognize text');
        })
        .finally(() => {
          Tesseract.terminate(); // Ensure Tesseract is terminated in all cases
        });
    }
  };

  const handleTextToSpeech = () => {
    const utterance = new SpeechSynthesisUtterance(translation);
    speechSynthesis.speak(utterance);
  };

  const removeFavourite = async () => {
    if (!translation) return;

    try {
      await removeFavoriteBasedOnInput(input); 
      setIsFavourite(false);
      //alert('Removed from favourites!');
    } catch (error) {
      console.error("Error removing document: ", error);
      alert('Failed to remove to favourites.');
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
                {profilePicUrl ? (
                    <img src={profilePicUrl} alt="User Icon" className="user-profile-pic" />
                ) : (
                    <div className="user-icon" />
                )}
                {showDropdown && <UserDropdown auth={auth} profilePicUrl={profilePicUrl} />}
      </div>
      <div className="translation-container">
        <div className="english-input-container">
          <label htmlFor="english">{translateToKlingon ? "English" : "Klingon"}</label>
          <textarea
            id="english"
            data-gramm="false"
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
          <input
            type="file"
            accept="image/*"
            capture="camera"
            onChange={handleImageUpload}
            style={{ display: 'none' }}
            id="camera-input"
            data-testid="camera-input"
          />
          <label htmlFor="camera-input" className="camera-button" data-testid="camera-button">
            <Icon icon={cameraIcon} className="camera-icon" />
          </label> 
          <button onClick={clearTextAreas} className="clear-button">
            <Icon icon={closeIcon} className="clear-button" />
          </button>
          </div>
        {input.trim() ? (
          <button className="translate-button" onClick={translateText}>
            <Icon icon={translateIcon} />
          </button>
        ) : (
          <button className="swap-button" onClick={toggleTranslationDirection}>
            <Icon icon={swapIcon} />
          </button>
        )}
        <div className="klingon-output-container">
          <label htmlFor="klingon">{translateToKlingon ? "Klingon" : "English"}</label>
          <textarea
            id="klingon"
            className="input"
            value={translating ? "Translating..." : translation}
            readOnly
            data-testid="output-textarea"
          />
          <button className="fav-button" onClick={isFavourite ? removeFavourite : handleFavourite} data-testid="fav-button">
            <Icon icon={heartIcon} className="fav-icon" style={{ color: isFavourite ? 'red' : 'black' }} />
          </button>
          <CopyToClipboard text={translation} onCopy={() => alert('Copied!')}>
            <Icon icon={copyIcon} className="copy-button" data-testid="copy-button" />
          </CopyToClipboard>
          <button className="speaker-button" onClick={handleTextToSpeech}>
            <Icon icon={speakerIcon} className="speaker-icon" />
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
