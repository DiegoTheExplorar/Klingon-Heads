import { signOut } from 'firebase/auth';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './UserDropdown.css';

function UserDropdown({ auth}) {
    const navigate = useNavigate();

    const handleSignOut = () => {
        signOut(auth).then(() => {
            navigate('/');
        }).catch((error) => {
            console.error('Error signing out: ', error);
        });
    };

    return (
        <div className="dropdown-menu">
            <button onClick={handleSignOut}>Sign Out</button>
            <button onClick={() => navigate('/translator')}>Translator</button>
            <button onClick={() => navigate('/learn')}>Learn</button>
            <button onClick={() => navigate('/quiz')}>Quiz</button>
        </div>
    );
}

export default UserDropdown;
