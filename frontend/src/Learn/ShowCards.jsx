import axios from 'axios';
import React, { useState } from 'react';
import '../UserDropdown.css';
import FlashCard from './FlashCard';
import InitialCard from './InitialCard';

function FetchDataComponent() {
    const [flashcard, setFlashcard] = useState(null);
    const [error, setError] = useState('');
    const [hasFetched, setHasFetched] = useState(false);


    const fetchFlashcard = async () => {
        try {
            const response = await axios.get('https://klingonapi-cafaedb94044.herokuapp.com/flashcard');
            setFlashcard(response.data);
            setError('');
            setHasFetched(true);
        } catch (err) {
            setError('Failed to fetch flashcard');
            console.error('API error:', err);
        }
    };


    return (
        <div>
            <div style={{ padding: '20px', textAlign: 'center' }}>
                {!hasFetched ? (
                    <InitialCard fetchFlashcard={fetchFlashcard} />
                ) : (
                    <>
                        <FlashCard flashcard={flashcard} />
                        <button onClick={fetchFlashcard} style={{ fontSize: '16px', padding: '10px', marginTop: '20px' }}>
                            Next Flashcard
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}

export default FetchDataComponent;
