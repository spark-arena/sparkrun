# from unittest.mock import patch
# from sparkrun.orchestration.sudo import run_indirect_sudo_script
# from sparkrun.orchestration.ssh import RemoteResult
# from sparkrun.utils.shell import b64_encode_cmd
#
#
# @patch("sparkrun.orchestration.ssh._run_subprocess")
# def test_run_indirect_sudo_script(mock_run):
#     mock_run.return_value = RemoteResult(host="1.2.3.4", returncode=0, stdout="ok", stderr="")
#     res = run_indirect_sudo_script(
#         host="1.2.3.4",
#         script="echo hello",
#         sudo_user="admin",
#         sudo_password="mypassword",
#         ssh_kwargs={"ssh_user": "bob"},
#     )
#     assert res.success
#     mock_run.assert_called_once()
#
#     args, kwargs = mock_run.call_args
#     # cmd = args[0]
#     # input_data = kwargs.get("input_data")
#
#     # assert "python3" in cmd
#     assert b64_encode_cmd("mypassword") in input_data
#     assert b64_encode_cmd("echo hello") in input_data
