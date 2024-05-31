import React, {FC, useState, SetStateAction} from 'react';

import ImputForm from '../../../components/form-control/InputFormComponent';
import Eye from '../../../../assets/eye.png';
import Email from '../../../../assets/email.png';
import LogoIcon from '../../../../assets/logo.png';
import UserItem from '../../../../assets/user-items.png';
import UserIcon from '../../../../assets/user-icon.png';

import {
  SafeView,
  Header,
  Logo,
  LogoImage,
  UserNav,
  UserImage,
  LoginContainer,
  Welcome,
  WelcomeText,
  WelcomeSloganText,
  BackgroundTop,
  BackgroundBottom,
  Group,
  ShowPassword,
  EyeIcon,
  IconDivider,
  SubmitButton,
  ButtonText,
  OnTouch,
} from '../components/login.styles';

interface Props {
  navigation: any;
}

const ResetPasswordScreen: FC<Props> = ({navigation}: Props) => {
  const [code, setCode] = useState<number | null>();
  const [newPassword, setNewPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [secureTextEntry, setSecureTextEntry] = useState<boolean>(true);
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
          <WelcomeText>Reset Your Password</WelcomeText>
          <WelcomeSloganText>Enter Your New Password Below</WelcomeSloganText>
        </Welcome>
        <Group>
          <ShowPassword>
            <EyeIcon source={JSON.parse(Email)} />
          </ShowPassword>
          <IconDivider />
          <ImputForm
            autoCapitalize="none"
            name="code"
            placeholder="Verification Code"
            value={code}
            onChangeText={(text: SetStateAction<number | null | undefined>) =>
              setCode(text)
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
            value={newPassword}
            onChangeText={(text: SetStateAction<string>) =>
              setNewPassword(text)
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
            name="confirmPassword"
            placeholder="Confirm New Password"
            secureTextEntry={secureTextEntry}
            value={confirmPassword}
            onChangeText={(text: SetStateAction<string>) =>
              setConfirmPassword(text)
            }
          />
        </Group>
        <OnTouch onPress={() => navigation.navigate('Login')}>
          <SubmitButton resizeMode="cover">
            <ButtonText>Change Password</ButtonText>
          </SubmitButton>
        </OnTouch>
      </LoginContainer>
      <BackgroundBottom />
    </SafeView>
  );
};

export default ResetPasswordScreen;
