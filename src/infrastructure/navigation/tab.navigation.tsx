/* eslint-disable react-native/no-inline-styles */
import React, {FC} from 'react';
import {Text, TouchableOpacity} from 'react-native';
import styled from 'styled-components/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';

import HomeStackNav from './home.stack.nav';
import AddJobItem from '../../features/add-job-item/screens/add-job.item';
import AccountMgmt from '../../features/account-mgmt/screens/account.mgmt';
import RemoveItem from '../../features/remove-item/screens/remove.item.screen';

import AccountIcon from '../../../assets/user-icon.png';
import UserAdd from '../../../assets/user-add.png';
import UserRemove from '../../../assets/user-remove.png';

const TabIconUser = styled.Image`
  width: 36px;
  height: 36px;
`;

const Tab = createBottomTabNavigator();

export const BackgroundBottom = styled.ImageBackground.attrs({
  source: require('../../../assets/bg-bottom.png'),
  resizeMode: 'contain',
})`
  right: -5px;
  bottom: -5px;
  position: absolute;
  z-index: -1;
`;

const TabBarBackground = styled.View`
  flex-direction: row;
  background-color: #3cb474;
  height: 80px;
  justify-content: center;
  align-items: center;
`;

interface Props {
  Home: string;
  AddItem: string;
  RemoveItem: string;
  AccountMgmt: string;
  navigation: any;
  state: any;
  descriptors: any;
}
const TAB_ICON: {[key: string]: any} = {
  AddItem: UserAdd,
  Home: UserRemove,
  RemoveItem: UserRemove,
  AccountMgmt: AccountIcon,
};

const CustomTabBar: FC<Props> = ({state, descriptors, navigation}) => {
  return (
    <>
      <TabBarBackground>
        {state.routes.map(
          (
            route: {key: string | number; name: string},
            index: React.Key | null | undefined,
          ) => {
            const {options} = descriptors[route.key];
            const label =
              options.tabBarLabel !== undefined
                ? options.tabBarLabel
                : options.title !== undefined
                ? options.title
                : route.name;
            const iconName = TAB_ICON[route.name];
            const isFocused = state.index === index;

            const onPress = () => {
              const event = navigation.emit({
                type: 'tabPress',
                target: route.key,
                canPreventDefault: true,
              });

              if (!isFocused && !event.defaultPrevented) {
                navigation.navigate({name: route.name, merge: true});
              }
              if (isFocused && route.name === 'RemoveItem') {
                navigation.navigate({name: 'Root', merge: true});
              }
            };

            const onLongPress = () => {
              navigation.emit({
                type: 'tabLongPress',
                target: route.key,
              });
            };

            return (
              <TouchableOpacity
                key={index}
                accessibilityRole="button"
                accessibilityState={isFocused ? {selected: true} : {}}
                accessibilityLabel={options.tabBarAccessibilityLabel}
                testID={options.tabBarTestID}
                onPress={onPress}
                onLongPress={onLongPress}
                style={{
                  flex: 1,
                  justifyContent: 'center',
                  alignItems: 'center',
                  display: route.name === 'Root' ? 'none' : 'flex',
                }}>
                <TabIconUser source={iconName} />
                <Text style={{color: isFocused ? '#EBBE7A' : '#fff'}}>
                  {label}
                </Text>
              </TouchableOpacity>
            );
          },
        )}
      </TabBarBackground>
    </>
  );
};

const TabNavigation = () => {
  return (
    <Tab.Navigator
      tabBar={props => (
        <CustomTabBar
          RemoveItem={''}
          Home={''}
          AddItem={''}
          AccountMgmt={''}
          {...props}
        />
      )}
      screenOptions={{
        headerShown: false,
        tabBarItemStyle: {
          backgroundColor: '#00ff00',
          margin: 5,
          borderRadius: 10,
          display: 'none',
        },
      }}
      initialRouteName="Root">
      <Tab.Screen
        name="Root"
        component={HomeStackNav}
        options={() => ({
          title: 'Items',
        })}
      />
      <Tab.Screen
        name="AddItem"
        component={AddJobItem}
        options={{
          tabBarLabel: 'Add item',
        }}
      />
      <Tab.Screen
        name="RemoveItem"
        component={RemoveItem}
        options={{
          tabBarLabel: 'Remove Item',
        }}
      />
      <Tab.Screen
        name="AccountMgmt"
        component={AccountMgmt}
        options={{
          tabBarLabel: 'Account Mgmt',
        }}
      />
    </Tab.Navigator>
  );
};

export default TabNavigation;
