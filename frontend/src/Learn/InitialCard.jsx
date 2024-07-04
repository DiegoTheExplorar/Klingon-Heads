import React, { useState } from 'react';
import './FlashCard.css'; // Make sure your CSS styles are set appropriately

function InitialCard({ flashcard, fetchFlashcard }) {
    const [isFlipped, setIsFlipped] = useState(false);

    const logoStyle = {
        width: '200px', // Control width here
        height: 'auto' // Maintain aspect ratio
    };
    
    // Function to get card style based on difficulty
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
            fetchFlashcard(); // Fetch the first flashcard if not already loaded
        } else {
            setIsFlipped(!isFlipped); // Flip the card if flashcard is loaded
        }
    };

    if (!flashcard) {
        // Show the logo card that can be clicked to load the first flashcard
        return (
            <div onClick={handleCardClick} className="card">
                <div className="card-front">
                    <img src="public/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" style={{ width: '50%', height: '50%' }} />
                </div>
            </div>
        );
    }

    // Return the actual flashcard once loaded
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
