import { getAuth } from "firebase/auth";
import { addDoc, collection, deleteDoc, doc, getDocs, query, where } from "firebase/firestore";
import { database } from "./firebaseConfig";


export async function addFavoriteToFirestore(input, translation) {
  const auth = getAuth(); 
  const currentUser = auth.currentUser; 

  if (!currentUser) {
    throw new Error('User not authenticated.'); 
  }

  const userFavoritesRef = collection(database, "users", currentUser.uid, "favourites");

  try {
    await addDoc(userFavoritesRef, {
      input: input.trim(),
      translation: translation,
      timestamp: new Date()
    });
  } catch (error) {
    throw new Error(`Failed to add to favourites: ${error.message}`);
  }
}

export async function addHistoryToFirestore(input, translation) {
  const auth = getAuth(); 
  const currentUser = auth.currentUser; 

  if (!currentUser) {
    throw new Error('User not authenticated.'); 
  }

  const userFavoritesRef = collection(database, "users", currentUser.uid, "history"); 

  try {
    await addDoc(userFavoritesRef, {
      input: input,
      translation: translation,
      timestamp: new Date() 
    });
  } catch (error) {
    throw new Error(`Failed to add to History: ${error.message}`);
  }
}


export async function getAllFavorites() {
    const auth = getAuth();
    const currentUser = auth.currentUser;

    if (!currentUser) {
        throw new Error('User not authenticated.');
    }

    const userFavoritesRef = collection(database, "users", currentUser.uid, "favourites");

    try {
        const snapshot = await getDocs(userFavoritesRef);
        return snapshot.docs.map(doc => ({
            id: doc.id, 
            ...doc.data() 
        }));
    } catch (error) {
        throw new Error(`Failed to retrieve favorites: ${error.message}`);
    }
}

export async function getHistory() {
  const auth = getAuth();
  const currentUser = auth.currentUser;

  if (!currentUser) {
      throw new Error('User not authenticated.');
  }

  const userFavoritesRef = collection(database, "users", currentUser.uid, "history");

  try {
      const snapshot = await getDocs(userFavoritesRef);
      return snapshot.docs.map(doc => ({
          id: doc.id, 
          ...doc.data() 
      }));
  } catch (error) {
      throw new Error(`Failed to retrieve history: ${error.message}`);
  }
}

export async function removeFavoriteFromFirestore(id) {
  const auth = getAuth();
  const currentUser = auth.currentUser;

  if (!currentUser) {
      throw new Error('User not authenticated.');
  }

  const userFavoritesRef = collection(database, 'users', currentUser.uid, 'favourites');
  const favoriteDocRef = doc(userFavoritesRef, id);

  try {
      await deleteDoc(favoriteDocRef);
  } catch (error) {
      throw new Error(`Failed to remove favorite: ${error.message}`);
  }
}


export async function checkFavoriteInFirestore(input) {
    const auth = getAuth();
    const currentUser = auth.currentUser;

    if (!currentUser) {
        throw new Error('User not authenticated.');
    }

    const userFavoritesRef = collection(database, 'users', currentUser.uid, 'favourites');
    
    const q = query(userFavoritesRef, where("input", "==", input.trim()));
    const querySnapshot = await getDocs(q);

    if (querySnapshot.empty) {
        console.log('No matching documents.');
        return false;
    } else {
        console.log('Document found:', querySnapshot.docs.map(doc => doc.data()));
        return true; 
    }
}

export async function removeFavoriteBasedOnInput(input) {
  const auth = getAuth();
  const currentUser = auth.currentUser;

  if (!currentUser) {
      throw new Error('User not authenticated.');
  }
  const userFavoritesRef = collection(database, 'users', currentUser.uid, 'favourites');
  const q = query(userFavoritesRef, where("input", "==", input.trim()));

  try {
      const querySnapshot = await getDocs(q);
      if (querySnapshot.empty) {
          console.log('No matching documents.');
          return false;
      }

      querySnapshot.forEach(async (docSnapshot) => {
          await deleteDoc(doc(userFavoritesRef, docSnapshot.id));
          console.log(`Deleted document with ID: ${docSnapshot.id}`);
      });

      return true;
  } catch (error) {
      console.error("Error removing document: ", error);
      throw error;
  }
}



