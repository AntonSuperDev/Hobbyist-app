/* eslint-disable react-hooks/exhaustive-deps */
import {
  BackgroundBottom,
  BackgroundTop,
  ButtonText,
  CheckboxContainer,
  CheckboxLabel,
  ErrorText,
  EyeIcon,
  ForgotPassword,
  ForgotPasswordText,
  Group,
  Header,
  IconDivider,
  Loading,
  LoginContainer,
  Logo,
  LogoImage,
  OnTouch,
  OnTouchSignUp,
  SafeView,
  ShowPassword,
  SignUpText,
  SubmitButton,
  TermAndContions,
  TermAndContionsText,
  TermAndContionsTextColor,
  UserImage,
  UserNav,
  Welcome,
  WelcomeSloganText,
  WelcomeText,
} from '../components/login.styles';
import React, {FC, useEffect, useState} from 'react';

import AsyncStorage from '@react-native-async-storage/async-storage';
import Email from '../../../../assets/email.png';
import Eye from '../../../../assets/eye.png';
import ImputForm from '../../../components/form-control/InputFormComponent';
import LogoIcon from '../../../../assets/logo.png';
// import {RootState} from '../../../store/store';
import {Switch} from 'react-native';
import UserIcon from '../../../../assets/user-icon.png';
import UserItem from '../../../../assets/user-items.png';
import { BASE_URL } from '../../../config';

// import {loginStart} from '../../../services/auth/auth.action';
// import {useDispatch} from 'react-redux';

interface Props {
  navigation: any;
  signIn: () => void;
}
const LoginScreen: FC<Props> = ({navigation}: Props) => {
  const [checked, setChecked] = useState(false);
  const [isLoadin, setIsLoadin] = useState(false);
  const [email, setEmail] = useState<string>('lincoln.smith.85@gmail.com');
  const [password, setPassword] = useState<string>('jZAMeEe90p2');
  const [secureTextEntry, setSecureTextEntry] = useState<boolean>(true);

  const [errorMsg, setErrorMsg] = useState(' ');
  //const {signIn} = useContext(AuthContext);

  // const dispatch = useDispatch();
  const checkLogin = async () => {
    const isLog = await AsyncStorage.getItem('savedLogin');
    // setIsLogin(isLog);
    if (isLog === 'true') {
      navigation.navigate('Home');
    }
    console.log('isLog', isLog);
  };
  const submitHandle = async () => {
    if (checked) {
      setIsLoadin(true);
      const response = await fetch(
        `${BASE_URL}/api/v1/signin`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/plain',
          },
          body: JSON.stringify({
            email: email,
            password: password,
          }),
        },
      );
      if (response.ok) {
        const res = await response.json();

        if (res.success) {
          // console.log(res);
          await AsyncStorage.setItem('token', res.data.token);
          await AsyncStorage.setItem('email', res.data.email);
          await AsyncStorage.setItem('name', res.data.fullname);
          await AsyncStorage.setItem('savedLogin', 'true');
          navigation.navigate('Home');
          setIsLoadin(false);
        } else {
          setErrorMsg(res.message);
          setIsLoadin(false);
        }
      }
    } else {
      setIsLoadin(true);
      const response = await fetch(
        `${BASE_URL}/api/v1/signin`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/plain',
          },
          body: JSON.stringify({
            email: email,
            password: password,
          }),
        },
      );
      if (response.ok) {
        const res = await response.json();

        if (res.success) {
          // console.log(res);
          await AsyncStorage.setItem('token', res.data.token);
          await AsyncStorage.setItem('email', res.data.email);
          await AsyncStorage.setItem('name', res.data.fullname);
          // await AsyncStorage.setItem('role', res.data.role);
          await AsyncStorage.setItem('savedLogin', 'false');
          navigation.navigate('Home');
          setIsLoadin(false);
        } else {
          setErrorMsg(res.message);
          setIsLoadin(false);
        }
      }
    }
  };
  useEffect(() => {
    checkLogin();
  }, []);
  const toggleSwitch = () => setChecked(previousState => !previousState);
  // const {isLoading, error} = useSelector((state: RootState) => state.user);

  return (
    <SafeView>
      <BackgroundTop />
      <LoginContainer>
        <Header>
          <Logo>
            <LogoImage source={JSON.parse(LogoIcon)} />
          </Logo>
          <UserNav>
            <UserImage source={JSON.parse(UserItem)} />
            <UserImage source={JSON.parse(UserIcon)} />
          </UserNav>
        </Header>
        <Welcome>
          <WelcomeText>WELCOME</WelcomeText>
          <WelcomeSloganText>
            Login to your account. Don't have account?{'\n'} Please{' '}
            <OnTouchSignUp onPress={() => navigation.navigate('Register')}>
              <SignUpText>Sign Up</SignUpText>
            </OnTouchSignUp>
          </WelcomeSloganText>
        </Welcome>
        {errorMsg !== '' ? <ErrorText>{errorMsg}</ErrorText> : null}

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
            onChangeText={(e: React.SetStateAction<string>) => setEmail(e)}
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
            name="unername"
            placeholder="Password"
            secureTextEntry={secureTextEntry}
            value={password}
            onChangeText={(text: React.SetStateAction<string>) =>
              setPassword(text)
            }
          />
        </Group>
        <CheckboxContainer>
          <Switch
            trackColor={{false: '#767577', true: '#81b0ff'}}
            thumbColor={checked ? '#f5dd4b' : '#f4f3f4'}
            ios_backgroundColor="#3e3e3e"
            onValueChange={toggleSwitch}
            value={checked}
          />
          <CheckboxLabel>Remember me?</CheckboxLabel>
        </CheckboxContainer>
        <ForgotPassword>
          <OnTouch onPress={() => navigation.navigate('ForgotPassord')}>
            <ForgotPasswordText>Forgot Password?</ForgotPasswordText>
          </OnTouch>
        </ForgotPassword>

        <OnTouch onPress={submitHandle}>
          <SubmitButton resizeMode="cover">
            {isLoadin === true ? (
              <Loading size={18} color="#000" />
            ) : (
              <ButtonText>Login</ButtonText>
            )}
          </SubmitButton>
        </OnTouch>

        <TermAndContions>
          <TermAndContionsText>
            Clicking on the button, you agree to the Hobbyist {'\n'}
            <TermAndContionsTextColor>
              Terms of Service
            </TermAndContionsTextColor>{' '}
            and
            <TermAndContionsTextColor> Privacy Policy</TermAndContionsTextColor>
          </TermAndContionsText>
        </TermAndContions>
      </LoginContainer>
      <BackgroundBottom />
    </SafeView>
  );
};

export default LoginScreen;
