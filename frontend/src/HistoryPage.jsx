import React, { useEffect, useState } from 'react';
import { getHistory } from './firebasehelper';

function HistoryPage() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        getHistory().then(hist => {
            setHistory(hist);
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
            <h2>Translation History</h2>
            <ul>
                {history.map((item, index) => (
                    <li key={index}>{item.input} - {item.translation}</li>
                ))}
            </ul>
        </div>
    );
}

export default HistoryPage;
