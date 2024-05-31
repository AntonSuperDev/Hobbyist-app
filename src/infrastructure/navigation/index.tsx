import React, {FC} from 'react';

// import AsyncStorage from '@react-native-async-storage/async-storage';
import ForgotPassordScreen from '../../features/auth/screens/forgot.password.screen';
// import HomeScreen from '../../features/Home/screens/home.screen';
import JobDetailScreen from '../../features/job-details/screens/job.detail.screen';
import LoginScreen from '../../features/auth/screens/login.screen';
// import AuthNavigator from './auth.navigator';
import {NavigationContainer} from '@react-navigation/native';
import ProfileScreen from '../../features/profile/screens/profile.screen';
import RegisterScreen from '../../features/auth/screens/register.screen';
import ResetPasswordScreen from '../../features/auth/screens/reset.password.screen';
// import {RootState} from '../../store/store';
import TabNavigation from './tab.navigation';
import {createStackNavigator} from '@react-navigation/stack';

// import {useSelector} from 'react-redux';

interface Props {
  user?: null;
}
// type RootStackProps = {
//   Login: undefined;
//   Register: undefined;
//   ForgotPassord: undefined;
//   ResetPassword: undefined;
// };
const Stack = createStackNavigator();
const Navigation: FC<Props> = () => {
  // const [currentUser, setCurrrentUser] = useState(undefined);
  // const {user} = useSelector((state: RootState) => state.user);

  return (
    <NavigationContainer>
      {/* {currentUser !== null || isLogin === 'true' ? ( */}
      {/* <Stack.Screen name="Root" component={AuthNavigator} /> */}
      {/* {currentUser !== null || isLogin === 'true' ? (
        <TabNavigation />
      ) : ( */}
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
        initialRouteName="Login">
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Register" component={RegisterScreen} />
        <Stack.Screen name="ForgotPassord" component={ForgotPassordScreen} />
        <Stack.Screen name="ResetPassword" component={ResetPasswordScreen} />
        {/* <Stack.Screen name="Home" component={LoginScreen} /> */}
        <Stack.Screen name="Home" component={TabNavigation} />
        <Stack.Screen name="Profile" component={ProfileScreen} />
        <Stack.Screen name="JobDetail" component={JobDetailScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default Navigation;
