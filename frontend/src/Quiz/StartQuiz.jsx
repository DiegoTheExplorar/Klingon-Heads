import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHighScoreFromFirestore } from '../FireBase/firebasehelper';
import './StartQuiz.css';

function StartQuiz() {
  const [showDropdown, setShowDropdown] = useState(false);
  const auth = getAuth();
  const [profilePicUrl, setProfilePicUrl] = useState(null);
  const navigate = useNavigate();
  const [highScores, setHighScores] = useState({
    english: null,
    klingon: null,
    random: null
  });

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      if (user) {
        setProfilePicUrl(user.photoURL);
        fetchHighScores();
      } else {
        setProfilePicUrl(null);
      }
    });
    return () => unsubscribe();
  }, [auth]);

  const fetchHighScores = async () => {
    try {
      const englishHighScore = await getHighScoreFromFirestore('english');
      const klingonHighScore = await getHighScoreFromFirestore('klingon');
      const randomHighScore = await getHighScoreFromFirestore('mixed');
      setHighScores({
        english: englishHighScore,
        klingon: klingonHighScore,
        random: randomHighScore
      });
    } catch (error) {
      console.error('Error fetching high scores:', error);
    }
  };

  return (
    <div className="start-quiz-container">
      <img src="/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" className="logo" />
      <h1 className="quiz-header">Quiz</h1>
      <p>Test your knowledge and see how much you can score!</p>
      <div className="quiz-options">
        <div className="quiz-button">
          <button onClick={() => navigate('/english-quiz')} className="start-quiz-button">English Quiz</button>
          {highScores.english && <p className="high-score">High Score: {highScores.english.score}</p>}
        </div>
        <div className="quiz-button">
          <button onClick={() => navigate('/klingon-quiz')} className="start-quiz-button">Klingon Quiz</button>
          {highScores.klingon && <p className="high-score">High Score: {highScores.klingon.score}</p>}
        </div>
        <div className="quiz-button">
          <button onClick={() => navigate('/random-quiz')} className="start-quiz-button">Random Quiz</button>
          {highScores.random && <p className="high-score">High Score: {highScores.random.score}</p>}
        </div>
      </div>
    </div>
  );
}

export default StartQuiz;
