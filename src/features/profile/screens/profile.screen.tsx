import {
  BackButton,
  BackgroundBottom,
  BackgroundBottomLeft,
  BackgroundTop,
  ButtonText,
  CurrentButton,
  CurrentButtonText,
  DeleteButton,
  EyeIcon,
  Group,
  IconDivider,
  ItemImage,
  OnTouch,
  ProfileContainer,
  ProfileTitle,
  ProfileTitleText,
  SafeView,
  ShowPassword,
  SubmitButton,
  TierBoxLeft,
  TierCollector,
  TierCollectorList,
  TierHeader,
  TierImage,
  TierLeftText,
  TierLevelIcon,
  TierLevelText,
  TierListItem,
  TierListItemText,
  Wrapper,
} from '../components/profile.styles';
/* eslint-disable react-native/no-inline-styles */
import React, {FC, useEffect, useState} from 'react';
import {Alert, Text, View} from 'react-native';

import AsyncStorage from '@react-native-async-storage/async-storage';
import Check from '../../../../assets/check.png';
import Email from '../../../../assets/email.png';
import {ErrorText} from '../../auth/components/login.styles';
import Eye from '../../../../assets/eye.png';
import HeaderContainer from '../../../components/header-components/header.component';
import ImputForm from '../../../components/form-control/InputFormComponent';
// import {useSelector} from 'react-redux';
import {Loading} from './../../auth/components/register.styles';
import Pouse from '../../../../assets/pouse.png';
import User from '../../../../assets/user.png';
import X from '../../../../assets/x.png';
import { BASE_URL } from '../../../config';

interface Props {
  navigation: any;
}
const ProfileScreen: FC<Props> = ({navigation}) => {
  const [fname, setFname] = useState<any | null>(null);
  const [email, setEmail] = useState<any | null>(null);
  const [initEmail, setInitEmail] = useState<any | null>(null);
  // const [user, setUser] = useState<string>('james-worker@gmail.com');
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [secureTextEntry, setSecureTextEntry] = useState<boolean>(true);
  const [isLoadin, setIsLoadin] = useState(false);
  const [errorMsg, setErrorMsg] = useState(' ');
  const [user, setUser] = useState('tier-I');

  // const {loginUser} = useSelector((state: RootState) => state.user);
  const getUser = async () => {
    const token = await AsyncStorage.getItem('token');
    var url =
      `${BASE_URL}/api/v1/get-user`;
    var headers = {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    };
    fetch(url, {
      method: 'GET',
      headers: headers,
    })
      .then(response => response.json())
      .then(response => {
        console.log(response);
        setUser(response.data.role);
        setFname(response.data.fullname);
        setEmail(response.data.email);
        setInitEmail(response.data.email);
      })
      .catch(err => {
        console.log(err);
      });
  };

  const deleteUser = async () => {
    setIsLoadin(true);
    const token = await AsyncStorage.getItem('token');
    var url =
      `${BASE_URL}/api/v1/delete-user`;
    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/plain',
        Authorization: 'Bearer ' + token,
      },
    });
    if (response.ok) {
      const res = await response.json();
      console.log(res);
      if (res.status === 200) {
        // console.log(res);
        navigation.navigate('Login');
        setIsLoadin(false);
      } else {
        setErrorMsg(res.message);
        setIsLoadin(false);
      }
    }
  };

  const updateUser = async () => {
    setIsLoadin(true);
    const token = await AsyncStorage.getItem('token');
    var url =
      `${BASE_URL}/api/v1/edit-user`;
    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/plain',
        Authorization: 'Bearer ' + token,
      },
      body: JSON.stringify({
        email: email,
        fullname: fname,
      }),
    });
    if (response.ok) {
      const res = await response.json();
      console.log(res);
      if (res.status === 200) {
        // console.log(res);
        setErrorMsg(res.message);
        if (initEmail !== email) {
          navigation.navigate('Login');
        } else {
          await AsyncStorage.setItem('email', email);
          await AsyncStorage.setItem('name', fname);
          setEmail(email);
          setFname(fname);
        }
        setIsLoadin(false);
      } else {
        setErrorMsg(res.message);
        setIsLoadin(false);
      }
    }
  };
  const showConfirmDialog = () => {
    return Alert.alert(
      'Are your sure?',
      'Are you sure you want to delete this account?',
      [
        {
          text: 'No',
        },
        {
          text: 'Yes',
          onPress: () => {
            deleteUser();
          },
        },
      ],
    );
  };

  useEffect(() => {
    getUser();
    Alert;
  }, []);
  return (
    <SafeView>
      <BackgroundTop />
      <Wrapper>
        <HeaderContainer navigation={navigation} />
      </Wrapper>
      <ProfileContainer>
        <ProfileTitle>
          <ProfileTitleText>Profile</ProfileTitleText>
          <OnTouch onPress={() => navigation.navigate('Home')}>
            <BackButton source={JSON.parse(X)} />
          </OnTouch>
        </ProfileTitle>
        {errorMsg !== '' ? <ErrorText>{errorMsg}</ErrorText> : null}
        <Group>
          <ShowPassword>
            <EyeIcon source={JSON.parse(User)} />
          </ShowPassword>
          <IconDivider />
          <ImputForm
            autoCapitalize="none"
            name="unername"
            placeholder="Full Name"
            value={fname}
            onChangeText={(text: React.SetStateAction<string>) =>
              setFname(text)
            }
          />
        </Group>
        <Group>
          <ShowPassword>
            <EyeIcon source={JSON.parse(Email)} />
          </ShowPassword>
          <IconDivider />
          <ImputForm
            autoCapitalize="none"
            name="email"
            placeholder="Email"
            value={email}
            onChangeText={(text: React.SetStateAction<string>) =>
              setEmail(text)
            }
          />
        </Group>
        <Group>
          {secureTextEntry ? (
            <ShowPassword onPress={() => setSecureTextEntry(false)}>
              <EyeIcon source={JSON.parse(Eye)} />
            </ShowPassword>
          ) : (
            <ShowPassword onPress={() => setSecureTextEntry(true)}>
              <EyeIcon source={JSON.parse(Eye)} />
            </ShowPassword>
          )}
          <IconDivider />
          <ImputForm
            autoCapitalize="none"
            name="newPassword"
            placeholder="New Password"
            secureTextEntry={secureTextEntry}
            value={password}
            onChangeText={(text: React.SetStateAction<string>) =>
              setPassword(text)
            }
          />
        </Group>
        <Group>
          {secureTextEntry ? (
            <ShowPassword onPress={() => setSecureTextEntry(false)}>
              <EyeIcon source={JSON.parse(Eye)} />
            </ShowPassword>
          ) : (
            <ShowPassword onPress={() => setSecureTextEntry(true)}>
              <EyeIcon source={JSON.parse(Eye)} />
            </ShowPassword>
          )}
          <IconDivider />
          <ImputForm
            autoCapitalize="none"
            name="confirmNewPassword"
            placeholder="Confirm New Password"
            secureTextEntry={secureTextEntry}
            value={confirmPassword}
            onChangeText={(text: React.SetStateAction<string>) =>
              setConfirmPassword(text)
            }
          />
        </Group>
        <OnTouch onPress={() => updateUser()}>
          <SubmitButton>
            {isLoadin === true ? (
              <Loading size={18} color="#000" />
            ) : (
              <ButtonText>Save Changes</ButtonText>
            )}
          </SubmitButton>
        </OnTouch>
        <OnTouch onPress={() => showConfirmDialog()}>
          <DeleteButton>
            {isLoadin === true ? (
              <Loading size={18} color="#000" />
            ) : (
              <ButtonText>Delete account</ButtonText>
            )}
          </DeleteButton>
        </OnTouch>
        <TierCollector>
          <TierHeader>
            {user === 'tier-I' ? (
              <TierBoxLeft>
                <TierLevelIcon>
                  <TierLevelText>I</TierLevelText>
                  <TierImage source={JSON.parse(Pouse)} />
                </TierLevelIcon>
                <TierLeftText>Tier Collector</TierLeftText>
              </TierBoxLeft>
            ) : null}
            {user === 'tier-II' ? (
              <TierBoxLeft>
                <TierLevelIcon>
                  <TierLevelText>II</TierLevelText>
                  <TierImage source={JSON.parse(Pouse)} />
                </TierLevelIcon>
                <TierLeftText>Tier Collector</TierLeftText>
              </TierBoxLeft>
            ) : null}
            {user === 'tier-III' ? (
              <TierBoxLeft>
                <TierLevelIcon>
                  <TierLevelText>III</TierLevelText>
                  <TierImage source={JSON.parse(Pouse)} />
                </TierLevelIcon>
                <TierLeftText>Tier Collector</TierLeftText>
              </TierBoxLeft>
            ) : null}
            <OnTouch onPress={() => null}>
              <CurrentButton>
                <CurrentButtonText>Current</CurrentButtonText>
              </CurrentButton>
            </OnTouch>
          </TierHeader>
        </TierCollector>
        {user === 'tier-I' ? (
          <TierCollectorList>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Allows entry of your collectibles into your hobbyist catalogue
              </TierListItemText>
            </TierListItem>
          </TierCollectorList>
        ) : null}
        {user === 'tier-II' ? (
          <TierCollectorList>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Daily pricing of your collectibles and total catalogue value
              </TierListItemText>
            </TierListItem>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Trending your collectibles value over time so you understand if
                it’s gaining or losing value (trend line does not start until
                you’ve entered item in app)
              </TierListItemText>
            </TierListItem>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Rapid linking to similar collectibles as your catalogue on
                e-commerce sites
              </TierListItemText>
            </TierListItem>
          </TierCollectorList>
        ) : null}
        {user === 'tier-III' ? (
          <TierCollectorList>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Daily pricing of your collectibles category so you can know if
                your collectible is outpacing its peer collectibles in a
                category
              </TierListItemText>
            </TierListItem>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Trending your collectibles category average price over time
              </TierListItemText>
            </TierListItem>
            <TierListItem>
              <ItemImage source={JSON.parse(Check)} />
              <TierListItemText>
                Rapid linking to collectibles on e-commerce sites that are
                different than your collectible but in the same category for
                catalogue optimization
              </TierListItemText>
            </TierListItem>
          </TierCollectorList>
        ) : null}
        <View
          style={{
            justifyContent: 'center',
            alignContent: 'center',
            alignItems: 'center',
            marginVertical: 20,
          }}>
          <OnTouch
            onPress={async () => {
              await AsyncStorage.setItem('savedLogin', 'false');
              await AsyncStorage.removeItem('token');
              navigation.navigate('Login');
            }}>
            <Text style={{color: 'red', fontSize: 20}}>Logout</Text>
          </OnTouch>
        </View>
        <BackgroundBottomLeft />
      </ProfileContainer>
      <BackgroundBottom />
    </SafeView>
  );
};

export default ProfileScreen;
