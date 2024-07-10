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
        <a href="#" onClick={() => { navigate('/'); closeMenu(); }}>Home</a>
        <a href="#" onClick={() => { navigate('/translator'); closeMenu(); }}>Translator</a>
        <a href="#" onClick={() => { navigate('/quiz'); closeMenu(); }}>Quiz</a>
        <a href="#" onClick={() => { navigate('/learn'); closeMenu(); }}>Learn</a>
        <button onClick={() => { handleSignOut(); closeMenu(); }} className="sign-out-button">Sign Out</button>
        {profilePicUrl && (
          <div className="profile-container">
            <img src={profilePicUrl} alt="Profile" className="profile-pic" />
            <span className="username">{username}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default HamburgerMenu;
