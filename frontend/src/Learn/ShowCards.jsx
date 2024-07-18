import axios from 'axios';
import React, { useState, useEffect } from 'react';
import './ShowCards.css'
import FlashCard from './FlashCard';
import InitialCard from './InitialCard';
import { Icon } from '@iconify/react';
import arrowRight from '@iconify-icons/mdi/arrow-right';
import heartIcon from '@iconify-icons/mdi/heart';
import { addFavoriteToFirestore, checkFavoriteInFirestore, removeFavoriteBasedOnInput } from '../FireBase/firebasehelper';


function FetchDataComponent() {
    const [flashcard, setFlashcard] = useState(null);
    const [error, setError] = useState('');
    const [hasFetched, setHasFetched] = useState(false);
    const [isFavourite, setIsFavourite] = useState(false);

    useEffect(() => {
        if (flashcard) {
            const checkFavouriteStatus = async () => {
                const exists = await checkFavoriteInFirestore(flashcard.english);
                setIsFavourite(exists);
            };
            checkFavouriteStatus();
        }
    }, [flashcard]);

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

    const handleFavourite = async () => {
        try {
            await addFavoriteToFirestore(flashcard.english, flashcard.klingon, "Klingon");
            setIsFavourite(true);
        } catch (error) {
            console.error("Error adding to favorites: ", error);
            alert('Failed to add to favourites.');
        }
    };

    const removeFavourite = async () => {
        try {
            await removeFavoriteBasedOnInput(flashcard.english);
            setIsFavourite(false);
        } catch (error) {
            console.error("Error removing from favorites: ", error);
            alert('Failed to remove from favourites.');
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
            {!hasFetched ? (
                <InitialCard fetchFlashcard={fetchFlashcard} />
            ) : (
                <>
                    <FlashCard flashcard={flashcard} handleFavourite={isFavourite ? removeFavourite : handleFavourite} isFavourite={isFavourite} />
                    <div className="button-container">
                        <button className="next-button" onClick={fetchFlashcard}>
                            <Icon icon={arrowRight} className="next-icon" />
                        </button>
                        <button className="fav-button" onClick={isFavourite ? removeFavourite : handleFavourite} data-testid="fav-button">
                            <Icon icon={heartIcon} className="fav-icon" style={{ color: isFavourite ? 'red' : 'black' }} />
                        </button>
                    </div>
                </>
            )}
        </div>
    );
};

export default FetchDataComponent;
