import React, { useState } from 'react';
import './FlashCard.css';

function InitialCard({ flashcard, fetchFlashcard }) {
    const [isFlipped, setIsFlipped] = useState(false);

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

    const handleCardClick = () => {
        if (!flashcard) {
            fetchFlashcard(); 
        } else {
            setIsFlipped(!isFlipped); 
        }
    };

    if (!flashcard) {
        return (
            <div onClick={handleCardClick} className="card">
                <div className="card-front">
                    <img src="public/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" style={{ width: '40%', height: '50%' }} />
                </div>
            </div>
        );
    }

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

export default InitialCard;
