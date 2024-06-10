import React, { useEffect, useState } from 'react';
import { getAllFavorites, removeFavoriteFromFirestore } from './firebasehelper';

function FavoritesComponent() {
    const [favorites, setFavorites] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        getAllFavorites().then(favs => {
            setFavorites(favs);
            setLoading(false);
        }).catch(err => {
            setError(err.message);
            setLoading(false);
        });
    }, []);

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
        <div>
            <h2>Your Favorites</h2>
            <ul>
                {favorites.map(fav => (
                    <li key={fav.id}>
                        {fav.input} - {fav.translation}
                        <button onClick={() => handleUnfavorite(fav.id)} className="unfavorite-button">
                            Unfavorite
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    );
}

export default FavoritesComponent;
