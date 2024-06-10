import React, { useEffect, useState } from 'react';
import { getAllFavorites } from './firebasehelper';

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

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div>
            <h2>Your Favorites</h2>
            <ul>
                {favorites.map(fav => (
                    <li key={fav.id}>{fav.input} - {fav.translation}</li>
                ))}
            </ul>
        </div>
    );
}

export default FavoritesComponent;
