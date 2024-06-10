import React from 'react';
import { Navigate } from 'react-router-dom';
import { auth } from './firebaseConfig';

const PrivateRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = React.useState(null);

  React.useEffect(() => {
    auth.onAuthStateChanged(user => {
      setIsAuthenticated(!!user);
    });
  }, []);

  if (isAuthenticated === null) {
    return <div>Loading...</div>; // or any loading indicator
  }

  return isAuthenticated ? children : <Navigate to="/auth" />;
};

export default PrivateRoute;
