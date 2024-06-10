import { signInWithPopup } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import Translator from './Translator';
import { auth, provider } from './config'; // Ensure path is correct

function SignIn() {
    const [email, setEmail] = useState('');

    useEffect(() => {
        const storedEmail = localStorage.getItem('email');
        if (storedEmail) {
            setEmail(storedEmail);
        }
    }, []);

    const handleSignIn = () => {
        signInWithPopup(auth, provider).then((result) => {
            const userEmail = result.user.email;
            setEmail(userEmail);
            localStorage.setItem('email', userEmail);
        }).catch((error) => {
            console.error('Error signing in: ', error);
        });
    };

    return (
        <div>
            {email ? <Translator /> : (
                <button onClick={handleSignIn}>Sign in with Google</button>
            )}
        </div>
    );
}

export default SignIn;
