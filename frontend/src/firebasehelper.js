import { getAuth } from "firebase/auth"; // Import the getAuth function to access Firebase authentication
import { addDoc, collection, deleteDoc, doc, getDocs, query, where, } from "firebase/firestore";
import { database } from "./firebaseConfig";


export async function addFavoriteToFirestore(input, translation) {
  const auth = getAuth(); // Get the auth object
  const currentUser = auth.currentUser; // Get the current authenticated user

  if (!currentUser) {
    throw new Error('User not authenticated.'); // Ensure user is authenticated
  }

  const userFavoritesRef = collection(database, "users", currentUser.uid, "favourites");

  try {
    await addDoc(userFavoritesRef, {
      input: input,
      translation: translation,
      timestamp: new Date()
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

export async function checkFavoriteInFirestore(input){
  console.log("Checking for favorite:", input);
  const auth = getAuth();
  const currentUser = auth.currentUser;

  if (!currentUser) {
      throw new Error('User not authenticated.');
  }

  console.log("Authenticated user ID:", currentUser.uid);
  const userFavoritesRef = collection(database, 'users', currentUser.uid, 'favourites');

  try {
    const q = query(userFavoritesRef, where("input", "==", input));
    const querySnapshot = await getDocs(q);
    console.log("Query Snapshot:", querySnapshot);
    console.log("Is snapshot empty:", querySnapshot.empty);
    return !querySnapshot.empty;

  } catch(error) {
    console.error("Error fetching favorites:", error);
    throw error;
  }
}


