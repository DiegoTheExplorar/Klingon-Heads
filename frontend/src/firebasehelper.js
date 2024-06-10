import { getAuth } from "firebase/auth"; // Import the getAuth function to access Firebase authentication
import { addDoc, collection, getDocs } from "firebase/firestore";
import { database } from "./firebaseConfig";


export async function addFavoriteToFirestore(input, translation) {
  const auth = getAuth(); // Get the auth object
  const currentUser = auth.currentUser; // Get the current authenticated user

  if (!currentUser) {
    throw new Error('User not authenticated.'); // Ensure user is authenticated
  }

  const userFavoritesRef = collection(database, "users", currentUser.uid, "favourites"); // Reference to the user's favorites collection

  try {
    await addDoc(userFavoritesRef, {
      input: input,
      translation: translation,
      timestamp: new Date() // Add a timestamp to each favorite
    });
  } catch (error) {
    throw new Error(`Failed to add to favourites: ${error.message}`);
  }
}

export async function addHistoryToFirestore(input, translation) {
  const auth = getAuth(); // Get the auth object
  const currentUser = auth.currentUser; // Get the current authenticated user

  if (!currentUser) {
    throw new Error('User not authenticated.'); // Ensure user is authenticated
  }

  const userFavoritesRef = collection(database, "users", currentUser.uid, "history"); // Reference to the user's history collection

  try {
    await addDoc(userFavoritesRef, {
      input: input,
      translation: translation,
      timestamp: new Date() // Add a timestamp to each translation
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
            id: doc.id, // include document ID if needed for reference or deletion
            ...doc.data() // spread the data in the document
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
          id: doc.id, // include document ID if needed for reference or deletion
          ...doc.data() // spread the data in the document
      }));
  } catch (error) {
      throw new Error(`Failed to retrieve history: ${error.message}`);
  }
}