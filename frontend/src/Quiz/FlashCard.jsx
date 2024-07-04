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
                return { backgroundColor: 'red' };
            default:
                return {};
        }
    };

    if (!flashcard) return null;

    return (
        <div onClick={handleCardClick} className={`card ${isFlipped ? 'flipped' : ''}`}>
            <div className="card-front" style={getCardStyle(flashcard.difficulty)}>
                <p>{flashcard.english}</p>
            </div>
            <div className="card-back" style={getCardStyle(flashcard.difficulty)}>
                <p>{flashcard.klingon}</p>
                <p>{flashcard.difficulty}</p>
            </div>
        </div>
    );
}

export default Flashcard;
