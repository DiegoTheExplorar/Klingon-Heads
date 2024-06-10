import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HistoryPage.css'; // Import the CSS file for styling
import { getHistory } from './firebasehelper';

function HistoryPage() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

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
        <div className="history-page">
            <h2>Translation History</h2>
            <button onClick={() => navigate('/translator')} className="navigate-button">
                        Go to Translator
            </button>
            {history.length === 0 ? (
                <div className="empty-message">
                    No translation history available. Please use the translator.
                </div>
            ) : (
                <table>
                    <thead>
                        <tr>
                            <th>Input</th>
                            <th>Translation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history.map((item, index) => (
                            <tr key={index}>
                                <td>{item.input}</td>
                                <td>{item.translation}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}

export default HistoryPage;
