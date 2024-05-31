import React, {FC} from 'react';

import ForgotPassordScreen from '../../features/auth/screens/forgot.password.screen';
import LoginScreen from '../../features/auth/screens/login.screen';
import RegisterScreen from '../../features/auth/screens/register.screen';
import ResetPasswordScreen from '../../features/auth/screens/reset.password.screen';
import {createStackNavigator} from '@react-navigation/stack';

type RootStackProps = {
  Login: undefined;
  Register: undefined;
  ForgotPassord: undefined;
  ResetPassword: undefined;
};
const Stack = createStackNavigator<RootStackProps>();

const AuthNavigator: FC = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: false,
    }}
    initialRouteName="Login">
    <Stack.Screen name="Login" component={LoginScreen} />
    <Stack.Screen name="Register" component={RegisterScreen} />
    <Stack.Screen name="ForgotPassord" component={ForgotPassordScreen} />
    <Stack.Screen name="ResetPassword" component={ResetPasswordScreen} />
  </Stack.Navigator>
);

export default AuthNavigator;
