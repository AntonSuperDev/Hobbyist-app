import {
  BackgroundBottom,
  BackgroundTop,
  ButtonText,
  EyeIcon,
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
} from '../components/register.styles';
import React, {FC, useState} from 'react';
import {useDispatch, useSelector} from 'react-redux';

import Email from '../../../../assets/email.png';
import {ErrorText} from '../components/login.styles';
import Eye from '../../../../assets/eye.png';
import ImputForm from '../../../components/form-control/InputFormComponent';
import LogoIcon from '../../../../assets/logo.png';
import {RootState} from '../../../store/store';
import User from '../../../../assets/user.png';
import UserIcon from '../../../../assets/user-icon.png';
import UserItem from '../../../../assets/user-items.png';
import {signUpStart} from '../../../services/auth/auth.action';

interface Props {
  navigation: any;
}

const RegisterScreen: FC<Props> = ({navigation}) => {
  const [fname, setFname] = useState<string>('');
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [secureTextEntry, setSecureTextEntry] = useState<boolean>(true);
  const dispatch = useDispatch();

  const {error, isLoading} = useSelector((state: RootState) => state.user);
  const submitHandle = async () => {
    if (password !== confirmPassword) {
      return;
    }
    dispatch(
      signUpStart({
        fullname: fname,
        email,
        password,
        c_password: confirmPassword,
      }),
    );
    if (!error) {
      navigation.navigate('Home');
    }
  };

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
          <WelcomeText>CREATE ACCOUNT</WelcomeText>
          <WelcomeSloganText>
            Please enter fields to start use the Hobbyist
          </WelcomeSloganText>
        </Welcome>
        {password !== confirmPassword && (
          <ErrorText>Password does not match</ErrorText>
        )}
        {error && <ErrorText>{error.message}</ErrorText>}
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
            onChangeText={(e: React.SetStateAction<string>) => setFname(e)}
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
            name="password"
            placeholder="Password"
            secureTextEntry={secureTextEntry}
            value={password}
            onChangeText={(e: React.SetStateAction<string>) => setPassword(e)}
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
            name="confirmPassword"
            placeholder="Confirm Password"
            secureTextEntry={secureTextEntry}
            value={confirmPassword}
            onChangeText={(e: React.SetStateAction<string>) =>
              setConfirmPassword(e)
            }
          />
        </Group>
        <OnTouch onPress={submitHandle}>
          <SubmitButton resizeMode="cover">
            {isLoading === true ? (
              <Loading size={18} color="#000" />
            ) : (
              <ButtonText>Create Account</ButtonText>
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
        <WelcomeSloganText>
          Return to{' '}
          <OnTouchSignUp onPress={() => navigation.goBack()}>
            <SignUpText>Log in</SignUpText>
          </OnTouchSignUp>
        </WelcomeSloganText>
      </LoginContainer>
      <BackgroundBottom />
    </SafeView>
  );
};

export default RegisterScreen;
