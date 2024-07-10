import removeIcon from '@iconify-icons/ic/twotone-close';
import arrowBack from '@iconify-icons/mdi/arrow-back';
import { Icon } from '@iconify/react';
import { getAuth } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAllFavorites, removeFavoriteFromFirestore } from '../FireBase/firebasehelper';
import './FavoritesPage.css';

function FavoritesPage() {
    const [favorites, setFavorites] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState('all');
    const navigate = useNavigate();
    const auth = getAuth();

    useEffect(() => {
        getAllFavorites().then(favs => {
            setFavorites(favs);
            setLoading(false);
        }).catch(err => {
            setError(err.message);
            setLoading(false);
        });
    }, []);



    const handleFilterChange = (newFilter) => {
        setFilter(newFilter);
    };

    const filteredFavorites = favorites.filter(item => {
        if (filter === 'all') return true;
        if (filter === 'englishToKlingon') return item.language === 'English';
        if (filter === 'klingonToEnglish') return item.language === 'Klingon';
        return false;
    });

    const handleUnfavorite = async (id) => {
        try {
            await removeFavoriteFromFirestore(id);
            setFavorites(prevFavorites => prevFavorites.filter(fav => fav.id !== id));
        } catch (error) {
            setError(error.message);
        }
    };

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div className="favorites-page">
            <button className="back-button" onClick={() => navigate('/translator')}>
                <Icon icon={arrowBack} className="back-icon" />
                Back to Translator
            </button>
            <img src="/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" className="logo" />
            <h2 className="favorites-header">Favorites</h2>
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
            {filteredFavorites.length === 0 ? (
                <div className="empty-message">
                    No favorites available.
                </div>
            ) : (
                <div className="favorites-table">
                    <div className="favorites-table-column">
                        {filteredFavorites.map((fav, index) => (
                            <div className="favorites-item" key={fav.id}>
                                <div className="favorites-input">{fav.input}</div>
                                <div className="favorites-output">{fav.translation}</div>
                                <button className="unfavorite-button" onClick={() => handleUnfavorite(fav.id)}>
                                    <Icon icon={removeIcon} className="unfavorite-icon" />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default FavoritesPage;
