import axios from 'axios';
import { getAuth, onAuthStateChanged, signOut } from 'firebase/auth';
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
            const response = await axios.get('http://127.0.0.1:5000/flashcard');
            setFlashcard(response.data);
            setError('');
            setHasFetched(true);
        } catch (err) {
            setError('Failed to fetch flashcard');
            console.error('API error:', err);
        }
    };

    const handleSignOut = () => {
        signOut(auth).then(() => {
            navigate('/'); // Redirect to homepage or login page after sign out
        }).catch((error) => {
            console.error('Error signing out: ', error);
        });
    };

    return (
        <div>
            <div style={{ padding: '20px', textAlign: 'center' }}>
                {!hasFetched ? (
                    // Show the initial card that fetches the first flashcard
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
