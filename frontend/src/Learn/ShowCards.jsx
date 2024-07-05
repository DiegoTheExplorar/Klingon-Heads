import axios from 'axios';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import UserDropdown from '../UserDropdown';
import '../UserDropdown.css';
import FlashCard from './FlashCard';
import InitialCard from './InitialCard';

function FetchDataComponent() {
    const [flashcard, setFlashcard] = useState(null);
    const [error, setError] = useState('');
    const [hasFetched, setHasFetched] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const auth = getAuth();
    const [profilePicUrl, setProfilePicUrl] = useState(null);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, user => {
            if (user) {
                setProfilePicUrl(user.photoURL);
            } else {
                setProfilePicUrl(null);
            }
        });
        return () => unsubscribe();
    }, [auth]);

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
            <div className="user-icon-container" onClick={() => setShowDropdown(!showDropdown)}>
                {profilePicUrl ? (
                    <img src={profilePicUrl} alt="User Icon" className="user-profile-pic" />
                ) : (
                    <div className="user-icon" />
                )}
                {showDropdown && <UserDropdown auth={auth} profilePicUrl={profilePicUrl} />}
            </div>
        </div>
    );
}

export default FetchDataComponent;
