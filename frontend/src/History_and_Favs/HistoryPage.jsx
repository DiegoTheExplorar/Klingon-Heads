import removeIcon from '@iconify-icons/ic/twotone-close';
import { Icon } from '@iconify/react';
import React, { useEffect, useState } from 'react';
import { getHistory, removeHistoryFromFirestore } from '../FireBase/firebasehelper';
import './HistoryPage.css';

function HistoryPage() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState('all'); // State to manage filter selection

    useEffect(() => {
        getHistory().then(hist => {
            setHistory(hist);
            setLoading(false);
        }).catch(err => {
            setError(err.message);
            setLoading(false);
        });
    }, []);

    const handleFilterChange = (newFilter) => {
        setFilter(newFilter);
    };

    const filteredHistory = history.filter(item => {
        if (filter === 'all') return true;
        if (filter === 'englishToKlingon') return item.language === 'English';
        if (filter === 'klingonToEnglish') return item.language === 'Klingon';

        return false;
    });

    const handleRemoveHistory = async (id) => {
        try {
            await removeHistoryFromFirestore(id);
            setHistory(prevHistory => prevHistory.filter(item => item.id !== id));
        } catch (error) {
            setError(error.message);
        }
    };

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div className="history-page">
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
                                <button onClick={() => handleRemoveHistory(item.id)} className="remove-button">
                                    <Icon icon={removeIcon} className="remove-icon" />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default HistoryPage;
