; ============================================================
; Inno Setup Script for Vision-Based Virtual Mouse
; ============================================================
; This script creates a professional Windows installer that:
;   - Installs to Program Files\VirtualMouse
;   - Creates Start Menu and Desktop shortcuts
;   - Registers uninstaller in Add/Remove Programs
;   - Bundles the hand_landmarker.task model
;   - Optionally launches the app after install
;
; Prerequisites:
;   1. Build the EXE first:  python build_exe.py
;   2. Install Inno Setup:  https://jrsoftware.org/isdl.php
;   3. Compile this script:  iscc installer\setup.iss
;      OR open it in the Inno Setup GUI and click Compile
;
; Output: installer\Output\VirtualMouseSetup.exe
; ============================================================

#define MyAppName "Virtual Mouse"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Sharjeel"
#define MyAppURL "https://github.com/sharjeelx03/vision-based-virtual-mouse"
#define MyAppExeName "VirtualMouse.exe"
#define MyAppDescription "Control your computer cursor with hand gestures"

[Setup]
; App identity
AppId={{B7E3A2F1-4D5C-4F8A-9B2E-1A3C5D7F9E0B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases

; Install paths
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output
OutputDir=Output
OutputBaseFilename=VirtualMouseSetup
SetupIconFile=..\assets\app_icon.ico

; Compression
Compression=lzma2
SolidCompression=yes

; Appearance
WizardStyle=modern
WizardSizePercent=120

; Privileges (user-level install by default)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Misc
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
VersionInfoDescription={#MyAppDescription}
VersionInfoVersion={#MyAppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupentry"; Description: "Start Virtual Mouse with Windows"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; Main application files (from PyInstaller dist output)
Source: "..\dist\VirtualMouse\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Model file (ensure it's included)
Source: "..\hand_landmarker.task"; DestDir: "{app}"; Flags: ignoreversion

; Logo (if exists)
Source: "..\logo.png"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyAppDescription}"

; Desktop shortcut (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyAppDescription}"

; Start Menu uninstall shortcut
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Registry]
; Start with Windows (optional task)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#MyAppName}"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: startupentry

[Run]
; Launch after install (optional)
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up settings on uninstall
Type: filesandordirs; Name: "{userappdata}\VirtualMouse"

[Code]
// Show a custom message on the welcome page
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
