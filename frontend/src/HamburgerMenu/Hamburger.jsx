import { faBook, faHistory, faHome, faLanguage, faQuestionCircle, faSignOutAlt, faStar } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { getAuth, onAuthStateChanged, signOut } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HamburgerMenu.css';

const HamburgerMenu = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [profilePicUrl, setProfilePicUrl] = useState(null);
  const [username, setUsername] = useState(null);
  const auth = getAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setProfilePicUrl(user.photoURL);
        setUsername(user.displayName);
      } else {
        setProfilePicUrl(null);
        setUsername(null);
        console.log("No user is signed in.");
      }
    });

    return () => unsubscribe();
  }, [auth]);

  const handleSignOut = () => {
    signOut(auth).then(() => {
      navigate('/');
    }).catch((error) => {
      console.error('Error signing out: ', error);
    });
  };

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const closeMenu = () => {
    setIsOpen(false);
  };

  return (
    <div>
      <div className={`menu-icon ${isOpen ? 'open' : ''}`} onClick={toggleMenu}>
        &#9776;
      </div>
      <div className={`menu ${isOpen ? 'open' : ''}`}>
        <button onClick={() => { navigate('/'); closeMenu(); }}>
          <FontAwesomeIcon icon={faHome} /> Home
        </button>
        <button onClick={() => { navigate('/translator'); closeMenu(); }}>
          <FontAwesomeIcon icon={faLanguage} /> Translator
        </button>
        <button onClick={() => { navigate('/quiz'); closeMenu(); }}>
          <FontAwesomeIcon icon={faQuestionCircle} /> Quiz
        </button>
        <button onClick={() => { navigate('/learn'); closeMenu(); }}>
          <FontAwesomeIcon icon={faBook} /> Learn
        </button>
        <button onClick={() => { navigate('/fav'); closeMenu(); }}>
          <FontAwesomeIcon icon={faStar} /> Favorites
        </button>
        <button onClick={() => { navigate('/history'); closeMenu(); }}>
          <FontAwesomeIcon icon={faHistory} /> History
        </button>
        <button className="sign-out-button" onClick={() => { handleSignOut(); closeMenu(); }}>
          <FontAwesomeIcon icon={faSignOutAlt} /> Sign Out
        </button>
        {profilePicUrl && (
          <div className="profile-container">
            <img src={profilePicUrl} alt="Profile" className="profile-pic" />
            <span className="username">{username}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default HamburgerMenu;
