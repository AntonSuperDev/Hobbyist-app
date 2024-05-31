import React, {FC, useState, SetStateAction} from 'react';

import ImputForm from '../../../components/form-control/InputFormComponent';
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
  TermAndContions,
  TermAndContionsText,
} from '../components/login.styles';

interface Props {
  navigation: any;
}

const ForgotPassordScreen: FC<Props> = ({navigation}: Props) => {
  const [email, setEmail] = useState<string | null>('');
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
          <WelcomeText>Emter Email Address Here</WelcomeText>
          <WelcomeSloganText>
            We'll send the verification code
          </WelcomeSloganText>
        </Welcome>
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
            onChangeText={(text: SetStateAction<string | null>) =>
              setEmail(text)
            }
          />
        </Group>
        <OnTouch onPress={() => navigation.navigate('ResetPassword')}>
          <SubmitButton resizeMode="cover">
            <ButtonText>Reset Password</ButtonText>
          </SubmitButton>
        </OnTouch>
        <TermAndContions>
          <TermAndContionsText>
            Forgot Your Password? Reset Here
          </TermAndContionsText>
        </TermAndContions>
      </LoginContainer>
      <BackgroundBottom />
    </SafeView>
  );
};

export default ForgotPassordScreen;
