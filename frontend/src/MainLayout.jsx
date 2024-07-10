import React from 'react';
import UserDropdown from './UserDropdown';

const MainLayout = ({ children }) => {
  return (
    <>
      <UserDropdown />
      <div>{children}</div>
    </>
  );
};

export default MainLayout;
