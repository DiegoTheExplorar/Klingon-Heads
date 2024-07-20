import React, { useState } from 'react';
import './FlashCard.css';

function Flashcard({ flashcard }) {
    const [isFlipped, setIsFlipped] = useState(false);

    const handleCardClick = () => {
        setIsFlipped(!isFlipped);
    };

    const getCardStyle = (difficulty) => {
        switch (difficulty) {
            case 'Easy':
                return { backgroundColor: 'lightgreen' };
            case 'Medium':
                return { backgroundColor: 'orange' };
            case 'Hard':
                return { backgroundColor: 'lightcoral' };
            default:
                return {};
        }
    };

    if (!flashcard) return null;

    return (
        <div>
          <h2 style={{fontSize: '24px', fontWeight: 'bold' , textAlign: 'center'}}>{flashcard.difficulty}</h2>
          <div onClick={handleCardClick} className={`card ${isFlipped ? 'flipped' : ''}`}>
            <div className="card-front" style={getCardStyle(flashcard.difficulty)}>
              <p>{flashcard.english}</p>
            </div>
            <div className="card-back" style={getCardStyle(flashcard.difficulty)}>
              <p>{flashcard.klingon}</p>
            </div>
          </div>
        </div>
      );
    }

export default Flashcard;
