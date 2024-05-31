import {ScrollView} from 'react-native';
import styled from 'styled-components/native';

export const OnScroll = styled(ScrollView).attrs({
  contentContainerStyle: {paddingBottom: 80},
})`
  height: 100%;
  background: rgba(255, 255, 255, 0.7);
  padding-bottom: 500px;
`;
export const AddItemContainer = styled.View`
  justify-items: center;
  padding-left: ${props => props.theme.space[6]}
  padding-right: ${props => props.theme.space[6]}
`;
export const ProfileTitle = styled.View`
  flex-direction: row;
  justify-content: space-between;
  padding-top: ${props => props.theme.space[4]}
  padding-bottom: ${props => props.theme.space[4]}
  border-bottom-width: 1px;
  border-bottom-color: #8ecfad;
  margin-bottom: ${props => props.theme.space[4]}
`;

export const ProfileTitleText = styled.Text`
  color: ${props => props.theme.colors.text.tertiary};
  font-size: ${props => props.theme.fontSizes.body}
  font-weight: ${props => props.theme.fontWeights.bold}
`;
export const BackButton = styled.Image`
  width: 24px;
  height: 24px;
`;

export const ImageUploadContainer = styled.View`
  border: 2px dashed #8ecfad;
  padding-top: ${props => props.theme.space[2]}
  padding-bottom: ${props => props.theme.space[2]}
  padding-left: ${props => props.theme.space[3]}
  padding-right: ${props => props.theme.space[3]}
  flex-direction: row;
  align-items: center;
`;
export const ImageUploadBox = styled.View`
  width: 51px;
  height: 51px;
  border-radius: 25px;
  background-color: ${props => props.theme.colors.bg.primary}
  padding: 8px;
  margin-right: ${props => props.theme.space[3]}
`;
export const ImageUpload = styled.Image`
  width: 35px;
  height: 35px;
  align-self: center;
  border-radius: 17px;
`;
export const ImageUploadText = styled.Text`
  color: ${props => props.theme.colors.text.tertiary};
  font-size: ${props => props.theme.fontSizes.subText};
  font-weight: ${props => props.theme.fontWeights.SemiBold};
`;

export const FormGroup = styled.View`
  background-color: ${props => props.theme.colors.bg.quinary};
  padding: ${props => props.theme.space[3]};
  border-radius: 2px;
`;
export const InputLabel = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.subText}
  font-weight: ${props => props.theme.fontWeights.bold}
  margin-bottom: ${props => props.theme.space[2]}
`;

export const Input = styled.TextInput.attrs({
  placeholderTextColor: '#8ECFAD',
})`
  width: 100%;
  min-height: ${props => props.theme.sizes[4]};
  padding-left: ${props => props.theme.space[3]};
  padding-right: ${props => props.theme.space[3]};
  padding-top: ${props => props.theme.space[2]}
  padding-bottom: ${props => props.theme.space[2]}
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid #3CB474;
  border-radius: 2px;
  font-size: ${props => props.theme.fontSizes.subText};
  color: ${props => props.theme.colors.text.primary};
`;

export const SubmitButton = styled.View`
    width: 100%;
    align-items: center;
    border-radius:4px;
    background-color: ${props => props.theme.colors.bg.secondary}
    padding-top: ${props => props.theme.space[3]}
    padding-bottom: ${props => props.theme.space[3]}
    margin-bottom: ${props => props.theme.space[3]};
`;
export const ButtonText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.body};
  font-weight: ${props => props.theme.fontWeights.bold};
`;
