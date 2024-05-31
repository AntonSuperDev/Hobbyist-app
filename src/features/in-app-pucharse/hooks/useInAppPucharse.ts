import AsyncStorage from '@react-native-async-storage/async-storage';
import {useEffect, useState} from 'react';
import {Platform} from 'react-native';
import * as Iap from 'react-native-iap';
import {SubscriptionPurchase, useIAP} from 'react-native-iap';
import { BASE_URL } from '../../../config';

// Play store item Ids
const productsIDs =
  Platform.select({
    android: ['tier_2_subscription', 'tier_3_subscription'],
    ios: ['tier__2', 'tier_3'],
  }) || [];
const tiersValues = ['tier-II', 'tier-III', 'tier-I'];

const subscriptionIdMap = {
  'tier-II': 0,
  'tier-III': 1,
  'tier-I': 2,
};

const iosSubscriptionsIdMap: any = {
  tier__2: 0,
  tier_3: 1,
};
const useInAppPurchase = () => {
  const [tierAvailable, setTierAvailable] = useState<number>(2);
  const [connectionErrorMsg, setConnectionErrorMsg] = useState('');
  const [loadingProducts, setLoadingProducts] = useState(true);
  const [selectedTier, setSelectedTier] = useState<number | null>(null);
  const [currentLoadingButton, setCurrentLoadingButton] = useState<
    number | null
  >(null);
  const [isRestoring, setIsRestoring] = useState(false);

  const {
    connected,
    subscriptions,
    getSubscriptions,
    finishTransaction,
    currentPurchaseError,
    requestSubscription,
    availablePurchases,
    getAvailablePurchases,
  } = useIAP();

  useEffect(() => {
    if (availablePurchases.length > 0) {
      const sortedAvailablePurchases = availablePurchases.sort(
        (a, b) => b.transactionDate - a.transactionDate,
      );

      checkCurrentPurchase(sortedAvailablePurchases[0], true);
    }
  }, [availablePurchases]);

  //Get products from play store.
  useEffect(() => {
    const gettingProducts = async () => {
      if (connected) {
        try {
          setLoadingProducts(true);
          getSubscriptions(productsIDs as string[]);
        } catch (error) {
          console.log(error);
        } finally {
          setLoadingProducts(false);
        }
      }
    };

    // const getCurrentRole = async () => {
    //   let currentRole = await AsyncStorage.getItem('role');
    //   !currentRole && (currentRole = 'tier-I');
    //   setTierAvailable(subscriptionIdMap[currentRole]);
    // };

    // getCurrentRole();
    gettingProducts();
  }, [connected, getSubscriptions]);

  const purchaseFullApp = async (productIndex: number | null) => {
    if (productIndex !== 0 && !productIndex) {
      return;
    }
    // Reset error msg
    if (connectionErrorMsg !== '') setConnectionErrorMsg('');
    if (!connected) {
      setConnectionErrorMsg('Please check your internet connection');
    }

    // If we are connected & have products, purchase the item. Google will handle if user has no internet here.
    else if (subscriptions?.length > 0) {
      try {
        const resp = await requestSubscription(productsIDs[productIndex]);
        checkCurrentPurchase(resp as SubscriptionPurchase);
      } catch (error) {
        console.log(error);
        setSelectedTier(null);
        setCurrentLoadingButton(null);
      }
    }
    // If we are connected but have no products returned, try to get products and purchase.
    else {
      try {
        await requestSubscription(productsIDs[productIndex]);
      } catch (error) {
        setSelectedTier(null);
        setCurrentLoadingButton(null);
      }
    }
  };

  const setTierRole = async (role: string) => {
    const token = await AsyncStorage.getItem('token');
    const apiUrl =
      `${BASE_URL}/api/v1/set-role`;

    const headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    await fetch(apiUrl, {
      method: 'PUT',
      headers: headers,
      body: JSON.stringify({role}),
    });
  };

  const checkCurrentPurchase = async (
    purchase?: Iap.SubscriptionPurchase,
    isRestoring: boolean = false,
  ) => {
    if (purchase && (selectedTier !== null || isRestoring)) {
      const receipt = purchase.transactionReceipt;
      if (receipt) {
        const tierValue =
          selectedTier !== null
            ? selectedTier
            : iosSubscriptionsIdMap[purchase.productId];
        // Give full app access
        setTierRole(tiersValues[tierValue as number]);
        setTierAvailable(tierValue as number);
        try {
          await finishTransaction(purchase);
        } catch (ackErr) {
          // We would need a backend to validate receipts for purhcases that pended for a while and were then declined. So I'll assume most purchase attempts go through successfully (OK ackResult) & take the hit for the ones that don't (user will still have full app access).
          console.log('ackError: ', ackErr);
        } finally {
          setSelectedTier(null);
          setCurrentLoadingButton(null);
          setIsRestoring(false);
        }
      } else {
        setSelectedTier(null);
        setCurrentLoadingButton(null);
      }
    } else {
      setSelectedTier(null);
      setCurrentLoadingButton(null);
    }
  };

  useEffect(() => {
    if (currentPurchaseError) {
      if (currentPurchaseError.code === 'E_ALREADY_OWNED') {
        setTierAvailable(selectedTier as number);
      }
    }
  }, [currentPurchaseError]);

  useEffect(() => {
    // Free tier
    if (selectedTier === 2) {
      setTierRole(tiersValues[2]);
      setTierAvailable(2);
      return;
    }
    if (selectedTier !== null) {
      setCurrentLoadingButton(selectedTier);
      purchaseFullApp(selectedTier);
    }
  }, [selectedTier]);

  const restoreSubscription = async () => {
    setIsRestoring(true);
    getAvailablePurchases();
  };

  return {
    subscriptions,
    loadingProducts,
    setSelectedTier,
    tierAvailable,
    currentLoadingButton,
    restoreSubscription,
    isRestoring,
  };
};
export default useInAppPurchase;
