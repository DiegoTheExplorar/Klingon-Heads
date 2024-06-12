import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HistoryPage.css'; // Import the CSS file for styling
import { getHistory } from './firebasehelper';
import { getAuth, onAuthStateChanged, signOut } from 'firebase/auth';
import { Icon } from '@iconify/react';
import arrowBack from '@iconify-icons/mdi/arrow-back';
import accountIcon from '@iconify-icons/mdi/account';

function HistoryPage() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState('all'); // State to manage filter selection
    const [showDropdown, setShowDropdown] = useState(false);
    const [profilePicUrl, setProfilePicUrl] = useState(null);
    const navigate = useNavigate();
    const auth = getAuth();

    useEffect(() => {
        getHistory().then(hist => {
            setHistory(hist);
            setLoading(false);
        }).catch(err => {
            setError(err.message);
            setLoading(false);
        });
    }, []);

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

    const handleSignOut = () => {
        signOut(auth).then(() => {
            navigate('/signin');
        }).catch((error) => {
            console.error('Error signing out: ', error);
        });
    };

    const handleFilterChange = (newFilter) => {
        setFilter(newFilter);
    };

    const filteredHistory = history.filter(item => {
        if (filter === 'all') return true;
        if (filter === 'englishToKlingon') return item.direction === 'englishToKlingon';
        if (filter === 'klingonToEnglish') return item.direction === 'klingonToEnglish';
        return false;
    });

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;
    
    return (
        <div className="history-page">
            <button className="back-button" onClick={() => navigate('/translator')}>
                <Icon icon={arrowBack} className="back-icon" />
                Back to Translator
            </button>
            <div className="user-icon-container" onClick={() => setShowDropdown(!showDropdown)}>
                {profilePicUrl ? (
                    <img src={profilePicUrl} alt="Profile" className="user-profile-pic" />
                ) : (
                    <Icon icon={accountIcon} className="user-icon" />
                )}
                {showDropdown && (
                    <div className="dropdown-menu">
                        <button onClick={handleSignOut}>Sign Out</button>
                        <button onClick={() => navigate('/profile')}>Profile</button>
                    </div>
                )}
            </div>
            <img src="/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" className="logo" />
            <h2 className="history-header">History</h2>
            <div className="filter-buttons">
                <button
                    className={filter === 'all' ? 'filter-button selected' : 'filter-button'}
                    onClick={() => handleFilterChange('all')}
                >
                    All
                </button>
                <button
                    className={filter === 'englishToKlingon' ? 'filter-button selected' : 'filter-button'}
                    onClick={() => handleFilterChange('englishToKlingon')}
                >
                    English to Klingon
                </button>
                <button
                    className={filter === 'klingonToEnglish' ? 'filter-button selected' : 'filter-button'}
                    onClick={() => handleFilterChange('klingonToEnglish')}
                >
                    Klingon to English
                </button>
            </div>
            {filteredHistory.length === 0 ? (
                <div className="empty-message">
                    No translation history available. Please use the translator.
                </div>
            ) : (
                <div className="history-table">
                    <div className="history-table-column">
                        {filteredHistory.map((item, index) => (
                            <div className="history-item" key={index}>
                                <div className="history-input">{item.input}</div>
                                <div className="history-output">{item.translation}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}


export default HistoryPage;
