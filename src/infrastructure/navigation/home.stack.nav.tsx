import React, {FC} from 'react';

import HomeScreen from '../../features/Home/screens/home.screen';
import JobDetailScreen from '../../features/job-details/screens/job.detail.screen';
import ProfileScreen from '../../features/profile/screens/profile.screen';
import {createStackNavigator} from '@react-navigation/stack';

type RootStackProps = {
  Home: undefined;
  Profile: undefined;
  JobDetail: any;
  // Login: any;
};
const Stack = createStackNavigator<RootStackProps>();

const HomeStackNav: FC = () => (
  <Stack.Navigator
    screenOptions={{
      headerShown: false,
    }}>
    <Stack.Screen name="Home" component={HomeScreen} />
    {/* <Stack.Screen name="Login" component={LoginScreen} /> */}
    <Stack.Screen name="Profile" component={ProfileScreen} />
    <Stack.Screen name="JobDetail" component={JobDetailScreen} />
  </Stack.Navigator>
);

export default HomeStackNav;
